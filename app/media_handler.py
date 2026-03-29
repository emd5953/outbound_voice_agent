# Media stream handler: bridges Twilio WebSocket with Deepgram STT and Cartesia TTS

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.call_orchestrator import CallOrchestrator

logger = logging.getLogger(__name__)


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
        """Continuously read Twilio WebSocket messages and forward audio.

        Runs as a long-lived coroutine.  Iterates over incoming Twilio
        messages, delegating each to :meth:`handle_twilio_message`.  Exits
        when a ``"stop"`` event is received or the WebSocket closes.
        """
        async for message in self.twilio_ws:
            data: dict = json.loads(message) if isinstance(message, (str, bytes)) else message
            await self.handle_twilio_message(
                message if isinstance(message, (str, bytes)) else json.dumps(message)
            )
            if data.get("event") == "stop":
                break

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
            await self.call_orchestrator.handle_transcript(transcript)

    async def process_deepgram_transcripts(self) -> None:
        """Continuously read Deepgram WebSocket messages and process transcripts.

        Runs as a long-lived coroutine.  Iterates over incoming Deepgram
        messages and delegates each to :meth:`handle_deepgram_transcript`.
        """
        async for message in self.deepgram_ws:
            await self.handle_deepgram_transcript(message)
