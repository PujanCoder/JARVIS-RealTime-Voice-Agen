"""Twilio Media Streams WebSocket adapter for the JARVIS orchestrator."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.agent.orchestrator import JarvisOrchestrator
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketLike(Protocol):
    """Small protocol compatible with FastAPI and common WebSocket servers."""

    async def receive_text(self) -> str: ...

    async def send_text(self, data: str) -> Any: ...

    async def close(self) -> Any: ...


class MediaTranscriber(Protocol):
    """Optional adapter for converting μ-law 8 kHz audio into recognized text."""

    async def transcribe(self, audio: bytes) -> str | None: ...


class AudioSynthesizer(Protocol):
    """Optional adapter returning μ-law 8 kHz audio suitable for Twilio."""

    async def synthesize(self, text: str) -> bytes: ...


class VoiceSession:
    """Maintain one call's stream identifier, agent state, and media adapters."""

    def __init__(
        self,
        orchestrator: JarvisOrchestrator | None = None,
        transcriber: MediaTranscriber | None = None,
        synthesizer: AudioSynthesizer | None = None,
        silence_frame_limit: int = 5,
    ) -> None:
        self.orchestrator = orchestrator or JarvisOrchestrator()
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.stream_sid: str | None = None
        self.media_frames = 0
        self._audio_buffer = bytearray()
        self._silent_frames = 0
        self._silence_frame_limit = max(1, silence_frame_limit)

    async def run(self, websocket: WebSocketLike) -> None:
        """Consume provider events until the call disconnects or stops."""
        try:
            while True:
                raw_event = await websocket.receive_text()
                if not await self.handle_event(websocket, raw_event):
                    return
        except (EOFError, ConnectionError, WebSocketDisconnect):
            logger.info("Voice WebSocket disconnected.")
        except Exception:
            logger.exception("Unexpected error in the voice WebSocket session.")
            await self._close_safely(websocket)

    async def handle_event(self, websocket: WebSocketLike, raw_event: str) -> bool:
        """Handle one Twilio event; return False when the session should stop."""
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed voice event.")
            return True
        if not isinstance(event, dict):
            logger.warning("Ignoring non-object voice event.")
            return True

        event_name = event.get("event")
        if event_name == "start":
            start = event.get("start")
            if isinstance(start, dict):
                stream_sid = start.get("streamSid")
                if isinstance(stream_sid, str):
                    self.stream_sid = stream_sid
            if self.stream_sid is None and isinstance(event.get("streamSid"), str):
                self.stream_sid = event["streamSid"]
            logger.info("Voice stream started.")
            return True
        if event_name == "stop":
            logger.info("Voice stream stopped after %d media frames.", self.media_frames)
            return False
        if event_name == "media":
            await self._handle_media(websocket, event)
            return True
        if event_name in {"transcript", "speech", "input", "user_input"}:
            text = self._extract_text(event)
            if text:
                await self._respond(websocket, text)
            else:
                logger.warning("Ignoring voice event without recognized text.")
            return True
        if event_name in {"connected", "mark"}:
            return True

        logger.warning("Ignoring unsupported voice event type %r.", event_name)
        return True

    async def _handle_media(self, websocket: WebSocketLike, event: dict[str, Any]) -> None:
        self.media_frames += 1
        media = event.get("media")
        payload = media.get("payload") if isinstance(media, dict) else None
        if not isinstance(payload, str):
            return
        try:
            audio = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            logger.warning("Ignoring malformed voice media payload.")
            return
        self._audio_buffer.extend(audio)
        if self.transcriber is None:
            return
        if self._is_silent(audio):
            self._silent_frames += 1
        else:
            self._silent_frames = 0
        if self._silent_frames < self._silence_frame_limit:
            return
        buffered_audio = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._silent_frames = 0
        try:
            text = await self.transcriber.transcribe(buffered_audio)
        except RuntimeError:
            logger.exception("Speech-to-text failed for voice utterance.")
            return
        if text and text.strip():
            await self._respond(websocket, text.strip())

    @staticmethod
    def _is_silent(audio: bytes, threshold: int = 3) -> bool:
        """Approximate silence detection for μ-law frames without decoding libraries."""
        if not audio:
            return True
        energy = 0.0
        for value in audio:
            decoded = (~value) & 0xFF
            magnitude = ((decoded & 0x0F) << 3) + 132
            magnitude <<= (decoded & 0x70) >> 4
            energy += abs(132 - magnitude if decoded & 0x80 else magnitude - 132)
        return energy / len(audio) <= threshold

    async def _respond(self, websocket: WebSocketLike, text: str) -> None:
        try:
            state = await self.orchestrator.handle(text)
        except RuntimeError:
            logger.exception("JARVIS agent failed for voice input.")
            return
        response = state.response
        if not response:
            return
        if self.synthesizer is None:
            logger.warning("No audio synthesizer configured; JARVIS response is not audible.")
            return
        try:
            audio = await self.synthesizer.synthesize(response)
        except RuntimeError:
            logger.exception("Text-to-speech failed for JARVIS response.")
            return
        if not isinstance(audio, bytes):
            raise TypeError("Audio synthesizer must return bytes.")
        if not self.stream_sid:
            logger.warning("Cannot send audio before the stream start event.")
            return
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(audio).decode("ascii")},
                }
            )
        )

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str | None:
        for key in ("text", "transcript", "utterance"):
            value = event.get(key)
            if isinstance(value, str):
                return value.strip()
        data = event.get("data")
        if isinstance(data, dict):
            for key in ("text", "transcript", "utterance"):
                value = data.get(key)
                if isinstance(value, str):
                    return value.strip()
        return None

    @staticmethod
    async def _close_safely(websocket: WebSocketLike) -> None:
        try:
            await websocket.close()
        except Exception:
            logger.debug("Voice WebSocket was already closed.", exc_info=True)


async def handle_voice_websocket(
    websocket: WebSocketLike,
    orchestrator: JarvisOrchestrator | None = None,
    transcriber: MediaTranscriber | None = None,
    synthesizer: AudioSynthesizer | None = None,
    silence_frame_limit: int = 5,
) -> None:
    """Convenience entry point for wiring a provider WebSocket route."""
    await VoiceSession(
        orchestrator,
        transcriber,
        synthesizer,
        silence_frame_limit,
    ).run(websocket)