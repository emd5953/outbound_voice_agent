# FastAPI server: /place-order endpoint, Twilio webhooks, WebSocket media stream

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from deepgram import AsyncDeepgramClient
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from pydantic import ValidationError
from twilio.rest import Client as TwilioClient

from app.call_orchestrator import CallOrchestrator
from app.conversation_engine import ConversationEngine
from app.media_handler import MediaStreamHandler
from app.models import CallResult, OrderContext, OrderPayload
from app.utils import extract_zip_code
from config import (
    BASE_URL,
    DEEPGRAM_API_KEY,
    RESTAURANT_PHONE,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Pizza Order Voice Agent")

# Active calls keyed by Twilio call_sid → CallOrchestrator
active_calls: dict[str, CallOrchestrator] = {}


class DeepgramSocketAdapter:
    """Thin adapter wrapping the Deepgram SDK's AsyncV1SocketClient.

    MediaStreamHandler expects a WebSocket-like object with:
    - ``send(bytes)`` to forward audio
    - ``async for message in ws`` yielding JSON strings

    The Deepgram SDK uses ``send_media(bytes)`` and ``recv()`` returning
    typed Pydantic objects.  This adapter bridges the two interfaces.
    """

    def __init__(self, dg_socket: Any) -> None:
        self._socket = dg_socket

    async def send(self, data: bytes) -> None:
        """Forward raw audio bytes to Deepgram."""
        await self._socket.send_media(data)

    def __aiter__(self) -> DeepgramSocketAdapter:
        return self

    async def __anext__(self) -> str:
        """Yield the next Deepgram message as a JSON string."""
        try:
            result = await self._socket.recv()
            # Convert Pydantic model to JSON string for MediaStreamHandler
            if hasattr(result, "model_dump_json"):
                return result.model_dump_json()
            if hasattr(result, "json"):
                return result.json()
            return json.dumps(result)
        except StopAsyncIteration:
            raise
        except Exception:
            raise StopAsyncIteration


@app.post("/place-order", response_model=CallResult)
async def place_order(payload: OrderPayload) -> CallResult:
    """Accept an order payload, initiate a Twilio outbound call, and return the result.

    - Validates the payload via Pydantic (automatic via type hint).
    - Extracts zip code from delivery_address.
    - Creates an OrderContext and CallOrchestrator.
    - Initiates a Twilio outbound call with webhook URL.
    - Stores the orchestrator in active_calls keyed by call_sid.
    - Awaits call completion and returns the CallResult.
    """
    # Extract zip code (raises ValueError if missing)
    try:
        zip_code = extract_zip_code(payload.delivery_address)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Build internal order context
    context = OrderContext(
        customer_name=payload.customer_name,
        phone_number=payload.phone_number,
        delivery_address=payload.delivery_address,
        zip_code=zip_code,
    )

    # Create the call orchestrator
    orchestrator = CallOrchestrator(order=payload, context=context)

    # Initiate Twilio outbound call
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    orchestrator.set_twilio_client(twilio_client)

    call = twilio_client.calls.create(
        to=RESTAURANT_PHONE,
        from_=TWILIO_FROM_NUMBER,
        url=f"{BASE_URL}/twilio/webhook",
        status_callback=f"{BASE_URL}/twilio/status",
    )

    # Store orchestrator keyed by call_sid
    orchestrator.call_sid = call.sid
    active_calls[call.sid] = orchestrator

    logger.info("Outbound call initiated: call_sid=%s", call.sid)

    # Wait for the call to complete and return the result
    result = await orchestrator.wait_for_completion()

    # Clean up
    active_calls.pop(call.sid, None)

    return result


# ------------------------------------------------------------------
# Twilio webhook: return TwiML connecting to WebSocket media stream
# ------------------------------------------------------------------


@app.post("/twilio/webhook")
async def twilio_webhook(request: Request) -> Response:
    """Return TwiML that connects the call to a WebSocket media stream.

    Twilio calls this URL when the outbound call is answered.  The TwiML
    instructs Twilio to open a bidirectional WebSocket to our /media-stream
    endpoint so we can send/receive audio.
    """
    host = request.headers.get("host", "")
    # Build a wss:// URL for the media stream WebSocket
    ws_url = f"wss://{host}/media-stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{ws_url}" />'
        "</Connect>"
        "</Response>"
    )

    return Response(content=twiml, media_type="application/xml")


# ------------------------------------------------------------------
# Twilio call status callback
# ------------------------------------------------------------------


@app.api_route("/twilio/status", methods=["GET", "POST"])
async def twilio_status(request: Request) -> Response:
    """Handle Twilio call status callbacks.

    Twilio sends POST requests with call status updates (ringing,
    in-progress, completed, failed, etc.).  We accept GET as well
    for flexibility.
    """
    if request.method == "POST":
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
    else:
        call_sid = request.query_params.get("CallSid", "")
        call_status = request.query_params.get("CallStatus", "")

    logger.info("Call status update: call_sid=%s status=%s", call_sid, call_status)

    # If the call ended externally, signal the orchestrator
    if call_status in ("completed", "failed", "busy", "no-answer", "canceled"):
        orchestrator = active_calls.get(call_sid)
        if orchestrator is not None and orchestrator.state.value != "hangup":
            await orchestrator.end_call("ivr_failure")

    return Response(content="<Response/>", media_type="application/xml")


# ------------------------------------------------------------------
# WebSocket media stream: Twilio bidirectional audio
# ------------------------------------------------------------------


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    """Accept Twilio WebSocket upgrade for bidirectional media streaming.

    On connect:
    1. Wait for Twilio 'start' event to get call_sid / stream_sid.
    2. Look up the CallOrchestrator from active_calls.
    3. Open a Deepgram STT WebSocket (mulaw, 8 kHz, mono).
    4. Create MediaStreamHandler, ConversationEngine, wire to orchestrator.
    5. Run two concurrent tasks via asyncio.gather:
       - forward_audio_to_deepgram (Twilio → Deepgram)
       - process_deepgram_transcripts (Deepgram → Orchestrator)
    6. On call end or error: close Deepgram and Twilio WebSockets,
       return partial results if available.

    Requirements: 14.1, 14.2, 14.3, 14.4, 16.3
    """
    await websocket.accept()
    logger.info("WebSocket /media-stream connected")

    call_sid: str = ""
    orchestrator: CallOrchestrator | None = None
    dg_connection: Any = None

    try:
        # ---- Phase 1: Wait for Twilio 'start' event ----
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            event = data.get("event", "")

            if event == "start":
                start_meta = data.get("start", {})
                call_sid = start_meta.get("callSid", "")
                stream_sid = start_meta.get("streamSid", "")
                logger.info(
                    "Media stream started: call_sid=%s stream_sid=%s",
                    call_sid,
                    stream_sid,
                )

                orchestrator = active_calls.get(call_sid)
                if orchestrator is None:
                    logger.error(
                        "No active orchestrator for call_sid=%s", call_sid
                    )
                    await websocket.close()
                    return

                orchestrator.stream_sid = stream_sid
                break

        # ---- Phase 2: Connect to Deepgram STT ----
        dg_client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
        dg_connection = dg_client.listen.v1.connect(
            model="nova-2",
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            interim_results=True,
            punctuate=True,
        )
        # The connect() call returns an async context manager
        dg_socket = await dg_connection.__aenter__()
        deepgram_ws = DeepgramSocketAdapter(dg_socket)

        # ---- Phase 3: Wire components together ----
        handler = MediaStreamHandler(
            twilio_ws=websocket,
            deepgram_ws=deepgram_ws,
            call_orchestrator=orchestrator,
        )
        handler.stream_sid = stream_sid
        orchestrator.set_media_handler(handler)

        engine = ConversationEngine(order=orchestrator.order)
        orchestrator.set_conversation_engine(engine)

        logger.info("All components wired for call_sid=%s", call_sid)

        # ---- Phase 4: Run concurrent audio pipelines ----
        await asyncio.gather(
            handler.forward_audio_to_deepgram(),
            handler.process_deepgram_transcripts(),
        )

    except Exception as exc:
        logger.error("WebSocket /media-stream error: %s", exc)
        # Graceful degradation: end call with partial results (Req 16.3)
        if orchestrator is not None and orchestrator.state.value != "hangup":
            await orchestrator.end_call("ivr_failure")
    finally:
        # ---- Phase 5: Cleanup ----
        if dg_connection is not None:
            try:
                await dg_connection.__aexit__(None, None, None)
            except Exception:
                logger.debug("Deepgram connection already closed")
        logger.info("WebSocket /media-stream closed: call_sid=%s", call_sid)
