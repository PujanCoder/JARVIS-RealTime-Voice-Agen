import base64
import asyncio
import json

from app.agent import JarvisBrain, JarvisOrchestrator
from app.voice.websocket import VoiceSession


class FakeWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self.events = [json.dumps(event) for event in events]
        self.sent: list[dict] = []
        self.closed = False

    async def receive_text(self) -> str:
        return self.events.pop(0)

    async def send_text(self, value: str) -> None:
        self.sent.append(json.loads(value))

    async def close(self) -> None:
        self.closed = True


class FakeProvider:
    async def complete(self, messages) -> str:
        return "I heard you."


class FakeTranscriber:
    async def transcribe(self, audio: bytes) -> str:
        assert audio
        return "What's on my schedule today?"


class FakeSynthesizer:
    async def synthesize(self, text: str) -> bytes:
        assert text == "I heard you."
        return b"\xff\x00"


def test_start_media_and_stop_events() -> None:
    speech_payload = base64.b64encode(b"\x00" * 160).decode()
    silence_payload = base64.b64encode(b"\xff" * 160).decode()
    websocket = FakeWebSocket(
        [
            {"event": "connected"},
            {"event": "start", "start": {"streamSid": "MZ_TEST"}},
            {"event": "media", "media": {"payload": speech_payload}},
            {"event": "media", "media": {"payload": silence_payload}},
            {"event": "media", "media": {"payload": silence_payload}},
            {"event": "stop"},
        ]
    )
    session = VoiceSession(
        orchestrator=JarvisOrchestrator(brain=JarvisBrain(FakeProvider())),
        transcriber=FakeTranscriber(),
        synthesizer=FakeSynthesizer(),
        silence_frame_limit=2,
    )
    asyncio.run(session.run(websocket))
    assert session.stream_sid == "MZ_TEST"
    assert session.media_frames == 3
    assert websocket.sent[0]["event"] == "media"
    assert websocket.sent[0]["streamSid"] == "MZ_TEST"


def test_stt_failure_does_not_crash_session() -> None:
    class BrokenTranscriber:
        async def transcribe(self, audio: bytes) -> str:
            raise RuntimeError("provider unavailable")

    speech_payload = base64.b64encode(b"\x00" * 160).decode()
    silence_payload = base64.b64encode(b"\xff" * 160).decode()
    websocket = FakeWebSocket(
        [
            {"event": "start", "start": {"streamSid": "MZ_TEST"}},
            {"event": "media", "media": {"payload": speech_payload}},
            {"event": "media", "media": {"payload": silence_payload}},
            {"event": "media", "media": {"payload": silence_payload}},
            {"event": "stop"},
        ]
    )
    session = VoiceSession(
        transcriber=BrokenTranscriber(),
        silence_frame_limit=1,
    )
    asyncio.run(session.run(websocket))
    assert websocket.closed is False


def test_tts_failure_does_not_crash_session() -> None:
    class BrokenSynthesizer:
        async def synthesize(self, text: str) -> bytes:
            raise RuntimeError("provider unavailable")

    class TextTranscriber:
        async def transcribe(self, audio: bytes) -> str:
            return "hello"

    speech_payload = base64.b64encode(b"\x00" * 160).decode()
    silence_payload = base64.b64encode(b"\xff" * 160).decode()
    websocket = FakeWebSocket(
        [
            {"event": "start", "start": {"streamSid": "MZ_TEST"}},
            {"event": "media", "media": {"payload": speech_payload}},
            {"event": "media", "media": {"payload": silence_payload}},
            {"event": "media", "media": {"payload": silence_payload}},
            {"event": "stop"},
        ]
    )
    session = VoiceSession(
        orchestrator=JarvisOrchestrator(brain=JarvisBrain(FakeProvider())),
        transcriber=TextTranscriber(),
        synthesizer=BrokenSynthesizer(),
        silence_frame_limit=2,
    )
    asyncio.run(session.run(websocket))
    assert websocket.sent == []
