"""Central request pipeline for the JARVIS agent."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from .brain import JarvisBrain
from .planner import JarvisPlanner
from .state import AgentState, PlannedAction

logger = logging.getLogger(__name__)
ToolExecutor = Callable[[PlannedAction], Awaitable[object]]


class JarvisOrchestrator:
    """Coordinate planning, reasoning, and state updates."""

    def __init__(
        self,
        brain: JarvisBrain | None = None,
        planner: JarvisPlanner | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.brain = brain or JarvisBrain.from_environment()
        self.planner = planner or JarvisPlanner()
        self.tool_executor = tool_executor
        self.state = AgentState()

    async def handle(self, user_message: str) -> AgentState:
        """Process a message and return the updated conversation state."""
        self.state.user_message = user_message
        self.state.response = None
        self.state.errors.clear()
        self.state.add_message("user", user_message)

        plan = self.planner.plan(user_message)
        self.state.planned_actions = [plan]
        if self.tool_executor and plan.requires_tool:
            try:
                result = await self.tool_executor(plan)
                self.state.metadata["last_tool_result"] = result
            except (RuntimeError, ValueError, TypeError):
                logger.exception("Tool executor failed for action %s.", plan.action)
                self.state.errors.append("The requested tool could not be completed.")

        history = self.state.conversation_history[:-1]
        self.state.response = await self.brain.respond(user_message, history, plan)
        self.state.add_message("assistant", self.state.response)
        return self.state

    async def respond(self, user_message: str) -> str:
        """Process a message and return only its assistant response."""
        state = await self.handle(user_message)
        return state.response or ""