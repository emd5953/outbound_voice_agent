"""Tests for MediaStreamHandler (task 7.1).

Covers Twilio message parsing (start/media/stop), audio forwarding to
Deepgram, and transcript processing with is_final filtering.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.media_handler import MediaStreamHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _twilio_start_msg(stream_sid: str = "MZ123") -> str:
    return json.dumps({"event": "start", "start": {"streamSid": stream_sid}})


def _twilio_media_msg(audio: bytes = b"\x00\x01\x02") -> str:
    payload = base64.b64encode(audio).decode()
    return json.dumps({"event": "media", "media": {"payload": payload}})


def _twilio_stop_msg() -> str:
    return json.dumps({"event": "stop"})


def _deepgram_final(transcript: str) -> str:
    return json.dumps(
        {
            "is_final": True,
            "channel": {"alternatives": [{"transcript": transcript}]},
        }
    )


def _deepgram_interim(transcript: str) -> str:
    return json.dumps(
        {
            "is_final": False,
            "channel": {"alternatives": [{"transcript": transcript}]},
        }
    )


def _make_handler(
    twilio_ws: Any = None,
    deepgram_ws: Any = None,
    orchestrator: Any = None,
) -> MediaStreamHandler:
    return MediaStreamHandler(
        twilio_ws=twilio_ws or AsyncMock(),
        deepgram_ws=deepgram_ws or AsyncMock(),
        call_orchestrator=orchestrator or AsyncMock(),
    )


# ---------------------------------------------------------------------------
# handle_twilio_message tests
# ---------------------------------------------------------------------------


class TestHandleTwilioMessage:
    @pytest.mark.asyncio
    async def test_start_event_extracts_stream_sid(self):
        handler = _make_handler()
        await handler.handle_twilio_message(_twilio_start_msg("MZ_ABC"))
        assert handler.stream_sid == "MZ_ABC"

    @pytest.mark.asyncio
    async def test_media_event_decodes_and_forwards_audio(self):
        deepgram_ws = AsyncMock()
        handler = _make_handler(deepgram_ws=deepgram_ws)
        raw_audio = b"\xff\xfe\xfd"
        await handler.handle_twilio_message(_twilio_media_msg(raw_audio))
        deepgram_ws.send.assert_awaited_once_with(raw_audio)

    @pytest.mark.asyncio
    async def test_stop_event_does_not_forward(self):
        deepgram_ws = AsyncMock()
        handler = _make_handler(deepgram_ws=deepgram_ws)
        await handler.handle_twilio_message(_twilio_stop_msg())
        deepgram_ws.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_event_is_ignored(self):
        deepgram_ws = AsyncMock()
        orchestrator = AsyncMock()
        handler = _make_handler(deepgram_ws=deepgram_ws, orchestrator=orchestrator)
        await handler.handle_twilio_message(json.dumps({"event": "mark"}))
        deepgram_ws.send.assert_not_awaited()
        orchestrator.handle_transcript.assert_not_awaited()


# ---------------------------------------------------------------------------
# handle_deepgram_transcript tests
# ---------------------------------------------------------------------------


class TestHandleDeepgramTranscript:
    @pytest.mark.asyncio
    async def test_final_non_empty_transcript_forwarded(self):
        orchestrator = AsyncMock()
        handler = _make_handler(orchestrator=orchestrator)
        await handler.handle_deepgram_transcript(_deepgram_final("hello there"))
        orchestrator.handle_transcript.assert_awaited_once_with("hello there")

    @pytest.mark.asyncio
    async def test_interim_transcript_ignored(self):
        orchestrator = AsyncMock()
        handler = _make_handler(orchestrator=orchestrator)
        await handler.handle_deepgram_transcript(_deepgram_interim("hello"))
        orchestrator.handle_transcript.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_final_empty_transcript_ignored(self):
        orchestrator = AsyncMock()
        handler = _make_handler(orchestrator=orchestrator)
        await handler.handle_deepgram_transcript(_deepgram_final(""))
        orchestrator.handle_transcript.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_final_whitespace_only_transcript_ignored(self):
        orchestrator = AsyncMock()
        handler = _make_handler(orchestrator=orchestrator)
        await handler.handle_deepgram_transcript(_deepgram_final("   "))
        orchestrator.handle_transcript.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_deepgram_message_ignored(self):
        orchestrator = AsyncMock()
        handler = _make_handler(orchestrator=orchestrator)
        # Missing channel/alternatives keys
        await handler.handle_deepgram_transcript(json.dumps({"is_final": True}))
        orchestrator.handle_transcript.assert_not_awaited()


# ---------------------------------------------------------------------------
# forward_audio_to_deepgram loop test
# ---------------------------------------------------------------------------


class TestForwardAudioLoop:
    @pytest.mark.asyncio
    async def test_loop_processes_start_media_stop(self):
        audio = b"\xaa\xbb"
        messages = [
            _twilio_start_msg("SID1"),
            _twilio_media_msg(audio),
            _twilio_stop_msg(),
        ]

        deepgram_ws = AsyncMock()

        # Make twilio_ws async-iterable over the messages list
        twilio_ws = AsyncMock()
        twilio_ws.__aiter__ = lambda self: self
        twilio_ws._iter = iter(messages)

        async def _anext(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

        twilio_ws.__anext__ = _anext

        handler = _make_handler(twilio_ws=twilio_ws, deepgram_ws=deepgram_ws)
        await handler.forward_audio_to_deepgram()

        assert handler.stream_sid == "SID1"
        deepgram_ws.send.assert_awaited_once_with(audio)


# ---------------------------------------------------------------------------
# process_deepgram_transcripts loop test
# ---------------------------------------------------------------------------


class TestProcessDeepgramLoop:
    @pytest.mark.asyncio
    async def test_loop_forwards_only_final_non_empty(self):
        messages = [
            _deepgram_interim("partial"),
            _deepgram_final(""),
            _deepgram_final("what can I get you"),
        ]

        orchestrator = AsyncMock()

        deepgram_ws = AsyncMock()
        deepgram_ws.__aiter__ = lambda self: self
        deepgram_ws._iter = iter(messages)

        async def _anext(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

        deepgram_ws.__anext__ = _anext

        handler = _make_handler(deepgram_ws=deepgram_ws, orchestrator=orchestrator)
        await handler.process_deepgram_transcripts()

        orchestrator.handle_transcript.assert_awaited_once_with("what can I get you")
