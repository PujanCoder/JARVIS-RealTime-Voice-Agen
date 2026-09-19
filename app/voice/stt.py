"""OpenAI-compatible speech-to-text adapter for Twilio μ-law media."""

from __future__ import annotations

import io
import os
import struct
import wave


class SpeechToTextError(RuntimeError):
    """Raised when speech transcription cannot be completed."""


def _mulaw_to_pcm(sample: int) -> int:
    sample = ~sample & 0xFF
    magnitude = ((sample & 0x0F) << 3) + 132
    magnitude <<= (sample & 0x70) >> 4
    return 132 - magnitude if sample & 0x80 else magnitude - 132


def mulaw_to_wav(audio: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap Twilio μ-law bytes as signed 16-bit PCM WAV for STT."""
    pcm = b"".join(struct.pack("<h", _mulaw_to_pcm(value)) for value in audio)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


class GroqSpeechTranscriber:
    """Transcribe buffered Twilio audio through Groq's Whisper endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        self.base_url = (base_url or os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise SpeechToTextError("GROQ_API_KEY is not configured.")

    async def transcribe(self, audio: bytes) -> str | None:
        """Submit one buffered utterance and return its transcript."""
        if not audio:
            return None
        try:
            import httpx
        except ImportError as exc:
            raise SpeechToTextError("httpx is required for speech transcription.") from exc

        files = {"file": ("twilio.wav", mulaw_to_wav(audio), "audio/wav")}
        data = {"model": self.model, "response_format": "text"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechToTextError("The speech-to-text request failed.") from exc
        transcript = response.text.strip()
        return transcript or None


# Backwards-compatible name for callers that injected the old generic adapter.
OpenAISpeechTranscriber = GroqSpeechTranscriber
