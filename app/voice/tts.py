"""OpenAI-compatible text-to-speech adapter with μ-law conversion."""

from __future__ import annotations

import io
import os
import struct
import wave


class TextToSpeechError(RuntimeError):
    """Raised when speech synthesis or audio conversion fails."""


def _linear_to_mulaw(sample: int) -> int:
    sample = max(-32768, min(32767, sample))
    sign = 0x80 if sample < 0 else 0
    magnitude = abs(sample)
    magnitude = min(magnitude, 32635) + 132
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (magnitude & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def wav_to_mulaw(audio: bytes) -> bytes:
    """Convert PCM WAV bytes at any sample rate to Twilio μ-law 8 kHz bytes."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            if channels not in (1, 2) or sample_width != 2 or sample_rate <= 0:
                raise TextToSpeechError("TTS audio must be mono/stereo 16-bit PCM.")
            pcm = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise TextToSpeechError("TTS provider returned invalid WAV audio.") from exc

    samples = [sample[0] for sample in struct.iter_unpack("<h", pcm)]
    if channels == 2:
        samples = [
            (samples[index] + samples[index + 1]) // 2
            for index in range(0, len(samples) - 1, 2)
        ]
    if sample_rate != 8000 and samples:
        output_length = max(1, round(len(samples) * 8000 / sample_rate))
        samples = [
            samples[min(len(samples) - 1, round(index * sample_rate / 8000))]
            for index in range(output_length)
        ]
    return bytes(_linear_to_mulaw(sample) for sample in samples)


class GroqTextSynthesizer:
    """Synthesize speech with Groq Orpheus and return Twilio μ-law 8 kHz audio."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
        self.voice = voice or os.getenv("TTS_VOICE", "troy")
        self.base_url = (base_url or os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise TextToSpeechError("GROQ_API_KEY is not configured.")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text as μ-law 8 kHz bytes."""
        if not text.strip():
            return b""
        try:
            import httpx
        except ImportError as exc:
            raise TextToSpeechError("httpx is required for speech synthesis.") from exc

        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/audio/speech",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TextToSpeechError("The text-to-speech request failed.") from exc
        return wav_to_mulaw(response.content)


# Backwards-compatible name for callers that injected the old generic adapter.
OpenAITextSynthesizer = GroqTextSynthesizer
