"""LLM abstraction and reasoning interface for JARVIS."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Protocol

from .prompts import (
    CLARIFICATION_PROMPT,
    DAILY_BRIEFING_PROMPT,
    ERROR_HANDLING_PROMPT,
    GENERAL_CONVERSATION_PROMPT,
    SYSTEM_PROMPT,
    TASK_PLANNING_PROMPT,
    build_user_prompt,
)
from .state import Message, PlannedAction

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Minimal provider contract, allowing any LLM SDK to be injected."""

    async def complete(self, messages: Sequence[Message]) -> str:
        """Return the assistant's text completion."""


class LLMConfigurationError(RuntimeError):
    """Raised when the configured provider cannot be used."""


class GroqLLMProvider:
    """Groq chat-completions provider using its OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
    ) -> None:
        if not api_key or not model:
            raise LLMConfigurationError("LLM_API_KEY and LLM_MODEL must be configured.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def complete(self, messages: Sequence[Message]) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise LLMConfigurationError(
                "The httpx dependency is required for the default LLM provider."
            ) from exc

        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("The configured LLM request failed.") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The LLM returned an unexpected response.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The LLM returned an empty response.")
        return content.strip()


# Backwards-compatible name for callers that injected the old generic provider.
HttpLLMProvider = GroqLLMProvider


class JarvisBrain:
    """Build prompts and obtain responses from an interchangeable LLM provider."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    @staticmethod
    def from_environment() -> "JarvisBrain":
        """Create a Groq-backed brain from environment variables."""
        api_key = os.getenv("LLM_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        model = os.getenv("LLM_MODEL", "")
        base_url = os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        )
        if not api_key or not model:
            return JarvisBrain()
        return JarvisBrain(GroqLLMProvider(api_key, model, base_url))

    async def respond(
        self,
        user_message: str,
        conversation_history: Sequence[Message] = (),
        planned_action: PlannedAction | None = None,
    ) -> str:
        """Generate a response, returning a safe user-facing error on provider failure."""
        if not user_message.strip():
            return "Please tell me what you would like help with."
        if self._provider is None:
            logger.error("No LLM provider configured; response cannot be generated.")
            return "I'm not configured with an LLM yet. Set LLM_API_KEY and LLM_MODEL, or inject an LLM provider."

        template = GENERAL_CONVERSATION_PROMPT
        if planned_action:
            template = {
                "briefing": DAILY_BRIEFING_PROMPT,
                "task": TASK_PLANNING_PROMPT,
                "unknown": CLARIFICATION_PROMPT,
            }.get(planned_action.intent, template)
        messages = [Message("system", SYSTEM_PROMPT)]
        messages.extend(conversation_history)
        messages.append(Message("user", build_user_prompt(user_message, template)))
        try:
            return await self._provider.complete(messages)
        except RuntimeError:
            logger.exception("LLM provider failed while generating a response.")
            return (
                "I'm sorry, I couldn't generate a response right now. "
                f"{ERROR_HANDLING_PROMPT.splitlines()[0]}"
            )