# Media stream handler: bridges Twilio WebSocket with Deepgram STT and Cartesia TTS

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import TYPE_CHECKING, Any

import cartesia

from config import CARTESIA_API_KEY

if TYPE_CHECKING:
    from app.call_orchestrator import CallOrchestrator

logger = logging.getLogger(__name__)

# Default Cartesia voice ID (Barbershop Man — natural male voice)
CARTESIA_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"
CARTESIA_MODEL_ID = "sonic"


class TTSError(Exception):
    """Raised when Cartesia TTS fails after all retry attempts."""


class MediaStreamHandler:
    """Bridges Twilio's bidirectional WebSocket media stream with Deepgram STT.

    Responsibilities handled by this class (task 7.1):
    - Parse Twilio WebSocket messages: "start", "media", "stop" events
    - Forward mulaw 8kHz audio from Twilio to Deepgram for transcription
    - Receive Deepgram transcripts and forward final, non-empty ones to the
      CallOrchestrator
    """

    def __init__(
        self,
        twilio_ws: Any,
        deepgram_ws: Any,
        call_orchestrator: CallOrchestrator,
    ) -> None:
        self.twilio_ws = twilio_ws
        self.deepgram_ws = deepgram_ws
        self.call_orchestrator = call_orchestrator
        self.stream_sid: str = ""

    # ------------------------------------------------------------------
    # Twilio → Deepgram: inbound audio forwarding
    # ------------------------------------------------------------------

    async def handle_twilio_message(self, message: str | bytes) -> None:
        """Parse a single Twilio WebSocket message and act on it.

        Twilio sends JSON messages with an ``event`` field:
        - ``"start"``  – stream metadata; extract ``streamSid``
        - ``"media"``  – base64-encoded mulaw audio; decode and forward to Deepgram
        - ``"stop"``   – stream ended; no further processing
        """
        data: dict = json.loads(message)
        event = data.get("event", "")

        if event == "start":
            self.stream_sid = data["start"]["streamSid"]
            logger.info("Twilio media stream started – streamSid=%s", self.stream_sid)

        elif event == "media":
            payload_b64: str = data["media"]["payload"]
            audio_bytes = base64.b64decode(payload_b64)
            await self.deepgram_ws.send(audio_bytes)

        elif event == "stop":
            logger.info("Twilio media stream stopped – streamSid=%s", self.stream_sid)

    async def forward_audio_to_deepgram(self) -> None:
        """Continuously read Twilio WebSocket messages and forward audio."""
        audio_count = 0
        try:
            while True:
                message = await self.twilio_ws.receive_text()
                data: dict = json.loads(message)
                await self.handle_twilio_message(message)
                if data.get("event") == "media":
                    audio_count += 1
                    if audio_count % 100 == 1:
                        logger.info("Audio chunks forwarded to Deepgram: %d", audio_count)
                if data.get("event") == "stop":
                    break
        except Exception as exc:
            logger.info("forward_audio_to_deepgram ended: %s", exc)

    # ------------------------------------------------------------------
    # Deepgram → Orchestrator: transcript processing
    # ------------------------------------------------------------------

    async def handle_deepgram_transcript(self, message: str | bytes) -> None:
        """Process a single Deepgram WebSocket message.

        Only acts on ``is_final`` results.  Extracts the top transcript
        alternative and, if non-empty, forwards it to the call orchestrator.
        """
        result: dict = json.loads(message)

        if not result.get("is_final", False):
            return

        try:
            transcript: str = result["channel"]["alternatives"][0]["transcript"]
        except (KeyError, IndexError):
            return

        transcript = transcript.strip()
        if transcript:
            logger.info("Deepgram transcript (is_final): %s", transcript)
            await self.call_orchestrator.handle_transcript(transcript)

    async def process_deepgram_transcripts(self) -> None:
        """Continuously read Deepgram WebSocket messages and process transcripts."""
        try:
            while True:
                message = await self.deepgram_ws.recv()
                # Debug: log raw Deepgram messages
                try:
                    parsed = json.loads(message)
                    msg_type = parsed.get("type", "unknown")
                    if msg_type == "Results":
                        is_final = parsed.get("is_final", False)
                        transcript = ""
                        try:
                            transcript = parsed["channel"]["alternatives"][0]["transcript"]
                        except (KeyError, IndexError):
                            pass
                        logger.info(
                            "Deepgram raw: type=%s is_final=%s transcript='%s'",
                            msg_type, is_final, transcript[:100],
                        )
                    else:
                        logger.info("Deepgram msg type=%s", msg_type)
                except Exception:
                    logger.info("Deepgram raw (unparsed): %s", str(message)[:200])
                await self.handle_deepgram_transcript(message)
        except Exception as exc:
            logger.info("process_deepgram_transcripts ended: %s", exc)


# ------------------------------------------------------------------
# Standalone TTS function: text → Cartesia → Twilio
# ------------------------------------------------------------------


async def synthesize_and_send(
    text: str,
    stream_sid: str,
    twilio_ws: Any,
    voice_id: str = CARTESIA_VOICE_ID,
) -> None:
    """Synthesize *text* via Cartesia TTS HTTP API and stream audio to Twilio.

    Uses the Cartesia REST API directly (bypassing the SDK which has Python 3.9
    compatibility issues) to generate raw pcm_mulaw at 8 kHz, then sends each
    chunk as a base64-encoded Twilio media event.

    Retry logic:
    - On first TTS failure, retry once after a 1-second backoff.
    - On second failure, raise TTSError.
    """
    import httpx

    max_attempts = 2
    backoff = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(
                    "https://api.cartesia.ai/tts/bytes",
                    headers={
                        "X-API-Key": CARTESIA_API_KEY,
                        "Cartesia-Version": "2024-06-10",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model_id": CARTESIA_MODEL_ID,
                        "transcript": text,
                        "voice": {"mode": "id", "id": voice_id},
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_mulaw",
                            "sample_rate": 8000,
                        },
                    },
                )
                response.raise_for_status()

                audio_bytes = response.content
                # Send in chunks to avoid huge single frames
                chunk_size = 640  # 80ms of mulaw at 8kHz
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i : i + chunk_size]
                    audio_b64 = base64.b64encode(chunk).decode("utf-8")
                    media_event = json.dumps(
                        {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_b64},
                        }
                    )
                    await twilio_ws.send_text(media_event)

            return

        except Exception as exc:
            if attempt < max_attempts:
                logger.warning(
                    "Cartesia TTS attempt %d failed (%s), retrying in %.1fs…",
                    attempt,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                logger.error(
                    "Cartesia TTS failed after %d attempts: %s",
                    max_attempts,
                    exc,
                )
                raise TTSError(
                    f"TTS synthesis failed after {max_attempts} attempts"
                ) from exc
