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


class _FakeResponse:
    """Mimics an AsyncBinaryAPIResponse with async iter_bytes."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class TestSynthesizeAndSend:
    @pytest.mark.asyncio
    async def test_streams_audio_chunks_as_base64_media_events(self):
        """Happy path: audio chunks are base64-encoded and sent to Twilio."""
        chunks = [b"\x01\x02", b"\x03\x04"]
        fake_response = _FakeResponse(chunks)

        mock_client = AsyncMock()
        mock_client.tts.generate = AsyncMock(return_value=fake_response)
        mock_client.close = AsyncMock()

        twilio_ws = AsyncMock()

        with patch("app.media_handler.cartesia.AsyncCartesia", return_value=mock_client):
            await synthesize_and_send("hello", "STREAM1", twilio_ws)

        # Verify generate was called with correct params
        mock_client.tts.generate.assert_awaited_once()
        call_kwargs = mock_client.tts.generate.call_args.kwargs
        assert call_kwargs["transcript"] == "hello"
        assert call_kwargs["output_format"]["encoding"] == "pcm_mulaw"
        assert call_kwargs["output_format"]["sample_rate"] == 8000
        assert call_kwargs["voice"] == {"mode": "id", "id": "a0e99841-438c-4a64-b679-ae501e7d6091"}

        # Verify each chunk was sent as a media event
        assert twilio_ws.send_text.await_count == 2
        for i, chunk in enumerate(chunks):
            sent_json = json.loads(twilio_ws.send_text.call_args_list[i].args[0])
            assert sent_json["event"] == "media"
            assert sent_json["streamSid"] == "STREAM1"
            expected_b64 = base64.b64encode(chunk).decode("utf-8")
            assert sent_json["media"]["payload"] == expected_b64

    @pytest.mark.asyncio
    async def test_retries_once_on_first_failure(self):
        """First TTS call fails, second succeeds — should still work."""
        fake_response = _FakeResponse([b"\xaa"])

        mock_client_fail = AsyncMock()
        mock_client_fail.tts.generate = AsyncMock(side_effect=RuntimeError("API down"))
        mock_client_fail.close = AsyncMock()

        mock_client_ok = AsyncMock()
        mock_client_ok.tts.generate = AsyncMock(return_value=fake_response)
        mock_client_ok.close = AsyncMock()

        twilio_ws = AsyncMock()

        with patch("app.media_handler.cartesia.AsyncCartesia", side_effect=[mock_client_fail, mock_client_ok]):
            with patch("app.media_handler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await synthesize_and_send("retry me", "SID2", twilio_ws)

        # Sleep was called once (backoff before retry)
        mock_sleep.assert_awaited_once_with(1.0)
        # Audio was sent on the second attempt
        assert twilio_ws.send_text.await_count == 1

    @pytest.mark.asyncio
    async def test_raises_tts_error_after_two_failures(self):
        """Both attempts fail — should raise TTSError."""
        mock_client = AsyncMock()
        mock_client.tts.generate = AsyncMock(side_effect=RuntimeError("API down"))
        mock_client.close = AsyncMock()

        twilio_ws = AsyncMock()

        with patch("app.media_handler.cartesia.AsyncCartesia", return_value=mock_client):
            with patch("app.media_handler.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TTSError, match="failed after 2 attempts"):
                    await synthesize_and_send("fail me", "SID3", twilio_ws)

        # No audio should have been sent
        twilio_ws.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_client_closed_on_success(self):
        """Client is closed after successful synthesis."""
        fake_response = _FakeResponse([b"\x00"])
        mock_client = AsyncMock()
        mock_client.tts.generate = AsyncMock(return_value=fake_response)
        mock_client.close = AsyncMock()

        twilio_ws = AsyncMock()

        with patch("app.media_handler.cartesia.AsyncCartesia", return_value=mock_client):
            await synthesize_and_send("hi", "SID4", twilio_ws)

        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_client_closed_on_failure(self):
        """Client is closed even when TTS fails."""
        mock_client = AsyncMock()
        mock_client.tts.generate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_client.close = AsyncMock()

        twilio_ws = AsyncMock()

        with patch("app.media_handler.cartesia.AsyncCartesia", return_value=mock_client):
            with patch("app.media_handler.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TTSError):
                    await synthesize_and_send("bye", "SID5", twilio_ws)

        # close() called on each attempt
        assert mock_client.close.await_count == 2

    @pytest.mark.asyncio
    async def test_custom_voice_id_passed_through(self):
        """A custom voice_id is forwarded to the Cartesia API."""
        fake_response = _FakeResponse([b"\x00"])
        mock_client = AsyncMock()
        mock_client.tts.generate = AsyncMock(return_value=fake_response)
        mock_client.close = AsyncMock()

        twilio_ws = AsyncMock()

        with patch("app.media_handler.cartesia.AsyncCartesia", return_value=mock_client):
            await synthesize_and_send("test", "SID6", twilio_ws, voice_id="custom-voice-123")

        call_kwargs = mock_client.tts.generate.call_args.kwargs
        assert call_kwargs["voice"] == {"mode": "id", "id": "custom-voice-123"}
