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

        # forward_audio_to_deepgram uses receive_text() in a while loop
        twilio_ws = AsyncMock()
        twilio_ws.receive_text = AsyncMock(side_effect=messages)

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

        # process_deepgram_transcripts uses recv() in a while loop
        deepgram_ws = AsyncMock()
        deepgram_ws.recv = AsyncMock(
            side_effect=[*messages, Exception("connection closed")]
        )

        handler = _make_handler(deepgram_ws=deepgram_ws, orchestrator=orchestrator)
        await handler.process_deepgram_transcripts()

        orchestrator.handle_transcript.assert_awaited_once_with("what can I get you")


# ---------------------------------------------------------------------------
# synthesize_and_send tests (task 7.2)
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock

from app.media_handler import synthesize_and_send, TTSError


async def _async_iter(items):
    """Helper: turn a list into an async iterator (for mocking aiter_lines)."""
    for item in items:
        yield item


def _make_sse_response(audio_data: bytes):
    """Build a mock streaming response that yields SSE lines with audio."""
    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    sse_lines = [f'data: {{"data": "{audio_b64}"}}', "data: [DONE]"]
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = lambda: _async_iter(sse_lines)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


class TestSynthesizeAndSend:
    @pytest.mark.asyncio
    async def test_streams_audio_chunks_as_base64_media_events(self):
        """Happy path: SSE audio chunks are base64-encoded and sent to Twilio."""
        audio_data = b"\x01\x02\x03\x04" * 200
        mock_response = _make_sse_response(audio_data)
        twilio_ws = AsyncMock()

        with patch("app.media_handler.httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.stream = MagicMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            await synthesize_and_send("hello", "STREAM1", twilio_ws)

        mock_client_instance.stream.assert_called_once()
        call_args = mock_client_instance.stream.call_args
        body = call_args.kwargs.get("json", {})
        assert body["transcript"] == "hello"
        assert body["output_format"]["encoding"] == "pcm_mulaw"
        assert body["output_format"]["sample_rate"] == 8000

        assert twilio_ws.send_text.await_count > 0
        for call in twilio_ws.send_text.call_args_list:
            sent_json = json.loads(call.args[0])
            assert sent_json["event"] == "media"
            assert sent_json["streamSid"] == "STREAM1"

    @pytest.mark.asyncio
    async def test_retries_once_on_first_failure(self):
        """First TTS call fails, second succeeds — should still work."""
        mock_response = _make_sse_response(b"\xaa" * 100)
        twilio_ws = AsyncMock()
        call_count = 0

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API down")
            return mock_response

        with patch("app.media_handler.httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.stream = MagicMock(side_effect=mock_stream)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            with patch("app.media_handler.asyncio.sleep", new_callable=AsyncMock):
                await synthesize_and_send("retry me", "SID2", twilio_ws)

        assert twilio_ws.send_text.await_count > 0

    @pytest.mark.asyncio
    async def test_raises_tts_error_after_two_failures(self):
        """Both attempts fail — should raise TTSError."""
        twilio_ws = AsyncMock()

        def mock_stream(*args, **kwargs):
            raise RuntimeError("API down")

        with patch("app.media_handler.httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.stream = MagicMock(side_effect=mock_stream)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            with patch("app.media_handler.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TTSError, match="failed after 2 attempts"):
                    await synthesize_and_send("fail me", "SID3", twilio_ws)

        twilio_ws.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_custom_voice_id_passed_through(self):
        """A custom voice_id is forwarded to the Cartesia API."""
        mock_response = _make_sse_response(b"\x00" * 100)
        twilio_ws = AsyncMock()

        with patch("app.media_handler.httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.stream = MagicMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            await synthesize_and_send("test", "SID6", twilio_ws, voice_id="custom-voice-123")

        call_args = mock_client_instance.stream.call_args
        body = call_args.kwargs.get("json", {})
        assert body["voice"]["id"] == "custom-voice-123"
