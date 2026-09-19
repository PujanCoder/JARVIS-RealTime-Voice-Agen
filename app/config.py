"""Environment-backed configuration for the JARVIS application."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or GROQ_API_KEY
LLM_MODEL = os.getenv("LLM_MODEL", "")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE", "troy")