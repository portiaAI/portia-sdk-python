"""Tests for the Storage classes."""

from pathlib import Path

import pytest

from portia.plan import Plan, PlanContext
from portia.plan_run import PlanRun, PlanRunState
from portia.storage import (
    DiskFileStorage,
    InMemoryStorage,
    PlanNotFoundError,
    PlanRunNotFoundError,
    PlanRunUUID,
    PlanUUID,
)


def test_in_memory_storage_save_and_get_plan() -> None:
    """Test saving and retrieving a Plan in InMemoryStorage."""
    storage = InMemoryStorage()
    plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
    storage.save_plan(plan)
    retrieved_plan = storage.get_plan(plan.id)

    assert retrieved_plan.id == plan.id

    with pytest.raises(PlanNotFoundError):
        storage.get_plan(PlanUUID())


def test_in_memory_storage_save_and_get_plan_run() -> None:
    """Test saving and retrieving PlanRun in InMemoryStorage."""
    storage = InMemoryStorage()
    plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
    plan_run = PlanRun(
        plan_id=plan.id,
        end_user_id="test123",
    )
    storage.save_plan_run(plan_run)
    retrieved_plan_run = storage.get_plan_run(plan_run.id)

    assert retrieved_plan_run.id == plan_run.id

    with pytest.raises(PlanRunNotFoundError):
        storage.get_plan_run(PlanRunUUID())


def test_disk_file_storage_save_and_get_plan(tmp_path: Path) -> None:
    """Test saving and retrieving a Plan in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    plan = Plan(plan_context=PlanContext(query="query", tool_ids=[]), steps=[])
    storage.save_plan(plan)
    retrieved_plan = storage.get_plan(plan.id)

    assert retrieved_plan.id == plan.id

    with pytest.raises(PlanNotFoundError):
        storage.get_plan(PlanUUID())


def test_disk_file_storage_save_and_get_plan_run(tmp_path: Path) -> None:
    """Test saving and retrieving PlanRun in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    plan = Plan(
        plan_context=PlanContext(query="query", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        end_user_id="test123",
    )
    storage.save_plan_run(plan_run)
    retrieved_plan_run = storage.get_plan_run(plan_run.id)

    assert retrieved_plan_run.id == plan_run.id

    with pytest.raises(PlanRunNotFoundError):
        storage.get_plan_run(PlanRunUUID())


def test_disk_file_storage_save_and_get_plan_runs(tmp_path: Path) -> None:
    """Test saving and retrieving PlanRun in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    plan = Plan(
        plan_context=PlanContext(query="query", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        state=PlanRunState.IN_PROGRESS,
        end_user_id="test123",
    )
    storage.save_plan_run(plan_run)
    plan_run = PlanRun(
        plan_id=plan.id,
        state=PlanRunState.FAILED,
        end_user_id="test123",
    )
    storage.save_plan_run(plan_run)

    runs = storage.get_plan_runs(PlanRunState.IN_PROGRESS)
    assert len(runs.results) == 1


def test_disk_file_storage_invalid_plan_retrieval(tmp_path: Path) -> None:
    """Test handling of invalid Plan data in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    invalid_file = tmp_path / "plan-invalid.json"
    invalid_file.write_text('{"id": "not-a-valid-uuid"}')  # Write invalid JSON

    with pytest.raises(PlanNotFoundError):
        storage.get_plan(PlanUUID())


def test_disk_file_storage_invalid_run_retrieval(tmp_path: Path) -> None:
    """Test handling of invalid Run data in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    invalid_file = tmp_path / "run-invalid.json"
    invalid_file.write_text('{"id": "not-a-valid-uuid"}')  # Write invalid JSON

    with pytest.raises(PlanRunNotFoundError):
        storage.get_plan_run(PlanRunUUID())
