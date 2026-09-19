"""FastAPI entrypoint and voice routes for the JARVIS application."""

from fastapi import FastAPI, HTTPException, Response, WebSocket
from dotenv import load_dotenv

from app.voice import (
    JarvisCaller,
    GroqSpeechTranscriber,
    GroqTextSynthesizer,
    VoiceConfigurationError,
    VoiceProviderError,
    handle_voice_websocket,
)

load_dotenv()

app = FastAPI(
    title="JARVIS",
    description="Personal AI assistant API",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict[str, str]:
    """Return the basic JARVIS service status."""
    return {"status": "online", "service": "JARVIS"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Return the application health status."""
    return {"status": "healthy"}


@app.post("/voice/stream", response_class=Response)
async def voice_stream() -> Response:
    """Return TwiML that connects an active call to the media WebSocket."""
    try:
        twiml = JarvisCaller().twiml_response()
    except VoiceConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/voice/ws")
async def voice_websocket(websocket: WebSocket) -> None:
    """Accept a Twilio Media Stream and connect it to JARVIS."""
    await websocket.accept()
    try:
        transcriber = GroqSpeechTranscriber()
        synthesizer = GroqTextSynthesizer()
    except (VoiceConfigurationError, RuntimeError) as exc:
        await websocket.close(code=1011, reason=str(exc))
        return
    await handle_voice_websocket(
        websocket,
        transcriber=transcriber,
        synthesizer=synthesizer,
    )


@app.post("/test-call")
async def test_call() -> dict[str, str]:
    """Place an outbound call only when explicitly requested."""
    try:
        call_sid = await JarvisCaller().create_outbound_call()
    except VoiceConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "calling", "call_sid": call_sid}