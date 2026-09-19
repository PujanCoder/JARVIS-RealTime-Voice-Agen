"""Manual smoke test for the app.agent implementation.

This script uses a local provider stub so it does not require API credentials,
network access, or any future JARVIS tools.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from app.agent import JarvisBrain, JarvisOrchestrator, JarvisPlanner
from app.agent.state import Message


class LocalTestProvider:
    """Small deterministic provider used to test the agent pipeline locally."""

    async def complete(self, messages: list[Message]) -> str:
        user_messages = [message.content for message in messages if message.role == "user"]
        latest_message = user_messages[-1].casefold() if user_messages else ""

        if "what project did i just mention" in latest_message:
            for message in reversed(user_messages[:-1]):
                if "main project is" in message.casefold():
                    return "You just mentioned your AI-powered travel planner."
            return "I do not have enough conversation context to identify the project."

        if "motivational briefing" in latest_message:
            return (
                "Today is a good day to make steady progress. Focus on one meaningful "
                "step, and let momentum build from there."
            )

        return "I understand. How would you like me to help?"


async def main() -> None:
    """Run planner, orchestrator, and conversation-memory smoke tests."""
    planner = JarvisPlanner()
    brain = JarvisBrain(provider=LocalTestProvider())
    orchestrator = JarvisOrchestrator(brain=brain, planner=planner)

    print("=== PLANNER TEST ===")
    plan = planner.plan("Remind me tomorrow to finish my MLOps project.")
    print(asdict(plan))

    print("\n=== ORCHESTRATOR TEST ===")
    briefing_state = await orchestrator.handle(
        "Hey JARVIS, give me a short motivational briefing for today."
    )
    print(briefing_state.response)

    print("\n=== MEMORY/CONTEXT TEST ===")
    await orchestrator.handle("My main project is an AI-powered travel planner.")
    context_state = await orchestrator.handle("What project did I just mention?")
    print(context_state.response)
    print(f"Messages preserved in state: {len(context_state.conversation_history)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"ERROR: Agent manual test failed: {exc}")
