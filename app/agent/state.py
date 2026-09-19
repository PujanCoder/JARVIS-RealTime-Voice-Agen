"""Shared state models for the JARVIS agent layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
Intent = Literal[
    "conversation", "email", "calendar", "task", "reminder", "briefing", "unknown"
]


@dataclass(slots=True)
class Message:
    """A single message exchanged with the agent or a connected tool."""

    role: Role
    content: str
    name: str | None = None


@dataclass(slots=True)
class PlannedAction:
    """A tool-independent description of an action the planner recommends."""

    intent: Intent
    action: str
    requires_tool: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None


@dataclass(slots=True)
class ToolResult:
    """The result of a future tool invocation."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None


@dataclass(slots=True)
class AgentState:
    """Mutable, serializable state passed through the agent pipeline."""

    user_message: str = ""
    conversation_history: list[Message] = field(default_factory=list)
    response: str | None = None
    current_task: str | None = None
    planned_actions: list[PlannedAction] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, role: Role, content: str, name: str | None = None) -> None:
        """Append a message and update the state timestamp."""
        self.conversation_history.append(Message(role=role, content=content, name=name))
        self.updated_at = datetime.now(timezone.utc)