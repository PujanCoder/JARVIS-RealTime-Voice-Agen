import asyncio

import app.main as main
from app.voice.call import JarvisCaller, TwilioSettings


def test_root_and_health() -> None:
    assert asyncio.run(main.root())["status"] == "online"
    assert asyncio.run(main.health())["status"] == "healthy"


def test_voice_stream_returns_twiml(monkeypatch) -> None:
    settings = TwilioSettings(
        "AC123",
        "secret",
        "+15551234567",
        "+15557654321",
        "https://example.ngrok-free.dev",
    )
    monkeypatch.setattr(main, "JarvisCaller", lambda: JarvisCaller(settings))
    response = asyncio.run(main.voice_stream())
    assert response.status_code == 200
    assert response.media_type == "application/xml"
    assert "wss://example.ngrok-free.dev/voice/ws" in response.body.decode()


def test_test_call_uses_mocked_twilio(monkeypatch) -> None:
    class FakeCaller:
        async def create_outbound_call(self) -> str:
            return "CA_TEST_ONLY"

    monkeypatch.setattr(main, "JarvisCaller", FakeCaller)
    response = asyncio.run(main.test_call())
    assert response == {"status": "calling", "call_sid": "CA_TEST_ONLY"}
