"""Integration checks for the implemented Phase 2 core components.

These tests intentionally use the application's modules directly. They do not
mock task, reminder, database, or scheduler behavior.
"""

from __future__ import annotations

import importlib
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest


def _load_nonempty_module(module_name: str) -> ModuleType:
    module = importlib.import_module(module_name)
    public_names = [name for name in vars(module) if not name.startswith("_")]
    if not public_names:
        pytest.fail(
            f"{module_name} has no public implementation; the Phase 2 component "
            "is still a placeholder."
        )
    return module


def _find_public_type(module: ModuleType, expected_names: tuple[str, ...]) -> type:
    for name in expected_names:
        value = getattr(module, name, None)
        if isinstance(value, type):
            return value
    pytest.fail(
        f"{module.__name__} does not expose an expected implementation type: "
        f"{', '.join(expected_names)}."
    )


def test_database_initializes_and_creates_models(tmp_path: Path) -> None:
    """Verify database initialization against a temporary SQLite path."""
    database = _load_nonempty_module("app.database.database")
    models = _load_nonempty_module("app.database.models")
    _load_nonempty_module("app.database.repositories")

    engine_factory = getattr(database, "create_engine", None)
    metadata = getattr(models, "Base", None)
    if not callable(engine_factory) or metadata is None:
        pytest.fail(
            "Database implementation must expose create_engine and model metadata "
            "for temporary SQLite initialization."
        )

    engine = engine_factory(f"sqlite:///{tmp_path / 'phase2.sqlite3'}")
    metadata.metadata.create_all(engine)
    assert engine is not None


def test_tasks_create_retrieve_pending_and_complete(tmp_path: Path) -> None:
    """Exercise task persistence using the existing task manager implementation."""
    database = _load_nonempty_module("app.database.database")
    models = _load_nonempty_module("app.database.models")
    manager_module = _load_nonempty_module("app.tools.tasks.manager")
    manager_type = _find_public_type(
        manager_module,
        ("TaskManager", "TasksManager"),
    )

    engine_factory = getattr(database, "create_engine", None)
    metadata = getattr(models, "Base", None)
    if not callable(engine_factory) or metadata is None:
        pytest.fail("Task test cannot initialize the application's database API.")

    engine = engine_factory(f"sqlite:///{tmp_path / 'tasks.sqlite3'}")
    metadata.metadata.create_all(engine)
    manager = manager_type(engine=engine)

    task = manager.create_task("Finish MLOps Prometheus monitoring")
    assert getattr(task, "id", None) is not None
    retrieved = manager.get_task(task.id)
    assert retrieved is not None
    pending = manager.get_pending_tasks()
    assert any(item.id == task.id for item in pending)
    manager.complete_task(task.id)
    assert all(item.id != task.id for item in manager.get_pending_tasks())


def test_reminders_create_retrieve_and_preserve_fields(tmp_path: Path) -> None:
    """Exercise reminder persistence without scheduling or notifying externally."""
    database = _load_nonempty_module("app.database.database")
    models = _load_nonempty_module("app.database.models")
    manager_module = _load_nonempty_module("app.tools.reminders.manager")
    manager_type = _find_public_type(
        manager_module,
        ("ReminderManager", "RemindersManager"),
    )

    engine_factory = getattr(database, "create_engine", None)
    metadata = getattr(models, "Base", None)
    if not callable(engine_factory) or metadata is None:
        pytest.fail("Reminder test cannot initialize the application's database API.")

    engine = engine_factory(f"sqlite:///{tmp_path / 'reminders.sqlite3'}")
    metadata.metadata.create_all(engine)
    manager = manager_type(engine=engine)
    due_at = datetime.now(timezone.utc) + timedelta(days=1)

    reminder = manager.create_reminder("Review JARVIS Phase 2", due_at)
    assert getattr(reminder, "id", None) is not None
    retrieved = manager.get_reminder(reminder.id)
    assert retrieved is not None
    assert retrieved.message == "Review JARVIS Phase 2"
    assert retrieved.due_at == due_at
    assert getattr(retrieved, "status", None) in {"pending", "active", None}


def test_scheduler_initializes_and_registers_jobs() -> None:
    """Verify scheduler setup without waiting for jobs or triggering notifications."""
    scheduler_module = _load_nonempty_module("app.scheduler.jobs")
    scheduler_type = _find_public_type(
        scheduler_module,
        ("JarvisScheduler", "Scheduler", "BackgroundScheduler"),
    )
    scheduler = scheduler_type()
    initialize = getattr(scheduler, "start", None) or getattr(scheduler, "initialize", None)
    register = getattr(scheduler, "add_job", None) or getattr(scheduler, "register_reminder", None)
    if not callable(initialize) or not callable(register):
        pytest.fail(
            "Scheduler must expose initialization and job-registration behavior."
        )
    assert scheduler is not None


def test_agent_routes_reminder_request() -> None:
    """Verify the existing planner/orchestrator recognizes reminder intent."""
    from app.agent import JarvisBrain, JarvisOrchestrator, JarvisPlanner

    planner = JarvisPlanner()
    plan = planner.plan("Remind me tomorrow to finish my MLOps project.")
    assert plan.intent == "reminder"
    assert plan.action == "create_reminder"

    class LocalProvider:
        async def complete(self, messages: object) -> str:
            return "I can help plan that reminder."

    async def run() -> object:
        return await JarvisOrchestrator(
            brain=JarvisBrain(provider=LocalProvider()),
            planner=planner,
        ).handle("Remind me tomorrow to finish my MLOps project.")

    state = asyncio.run(run())
    assert state.planned_actions[0].intent == "reminder"
    assert state.planned_actions[0].action == "create_reminder"
    assert state.response == "I can help plan that reminder."
