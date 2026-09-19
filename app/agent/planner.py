"""Lightweight intent and action planning for JARVIS."""

from __future__ import annotations

import re
from typing import Any

from .state import Intent, PlannedAction


class JarvisPlanner:
    """Classify requests without invoking external tools or services."""

    _patterns: tuple[tuple[Intent, str, tuple[str, ...]], ...] = (
        ("email", "manage_email", ("email", "inbox", "gmail", "send a mail")),
        ("calendar", "manage_calendar", ("calendar", "meeting", "appointment", "schedule")),
        ("task", "create_task", ("task", "todo", "to-do", "remember to")),
        ("reminder", "create_reminder", ("remind me", "reminder", "don't let me forget")),
        ("briefing", "daily_briefing", ("daily briefing", "morning briefing", "what's on today")),
    )

    def plan(self, user_message: str) -> PlannedAction:
        """Return a deterministic plan for a user message."""
        text = user_message.strip()
        normalized = text.casefold()
        if not text:
            return PlannedAction("unknown", "clarify_request", rationale="The message is empty.")

        for intent, action, keywords in self._patterns:
            if any(keyword in normalized for keyword in keywords):
                return PlannedAction(
                    intent=intent,
                    action=action,
                    requires_tool=True,
                    parameters=self._extract_parameters(text, intent),
                )

        if re.search(r"\b(what|who|why|how|can you|tell me|explain)\b", normalized):
            return PlannedAction("conversation", "respond", parameters={"topic": text})
        return PlannedAction("unknown", "clarify_or_respond", parameters={"request": text})

    @staticmethod
    def _extract_parameters(text: str, intent: Intent) -> dict[str, Any]:
        """Capture the original request without pretending to parse tool schemas."""
        return {"request": text, "category": intent}