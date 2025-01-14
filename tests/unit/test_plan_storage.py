"""Tests for the Storage classes."""

from pathlib import Path
from uuid import uuid4

import pytest

from portia.plan import Plan, PlanContext
from portia.storage import (
    DiskFileStorage,
    InMemoryStorage,
    PlanNotFoundError,
    WorkflowNotFoundError,
)
from portia.workflow import Workflow


def test_in_memory_storage_save_and_get_plan() -> None:
    """Test saving and retrieving a Plan in InMemoryStorage."""
    storage = InMemoryStorage()
    plan = Plan(plan_context=PlanContext(query="query", tool_list=[]), steps=[])
    storage.save_plan(plan)
    retrieved_plan = storage.get_plan(plan.id)

    assert retrieved_plan.id == plan.id

    with pytest.raises(PlanNotFoundError):
        storage.get_plan(uuid4())


def test_in_memory_storage_save_and_get_workflow() -> None:
    """Test saving and retrieving a Workflow in InMemoryStorage."""
    storage = InMemoryStorage()
    plan = Plan(plan_context=PlanContext(query="query", tool_list=[]), steps=[])
    workflow = Workflow(plan_id=plan.id)
    storage.save_workflow(workflow)
    retrieved_workflow = storage.get_workflow(workflow.id)

    assert retrieved_workflow.id == workflow.id

    with pytest.raises(WorkflowNotFoundError):
        storage.get_workflow(uuid4())


def test_disk_file_storage_save_and_get_plan(tmp_path: Path) -> None:
    """Test saving and retrieving a Plan in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    plan = Plan(plan_context=PlanContext(query="query", tool_list=[]), steps=[])
    storage.save_plan(plan)
    retrieved_plan = storage.get_plan(plan.id)

    assert retrieved_plan.id == plan.id

    with pytest.raises(PlanNotFoundError):
        storage.get_plan(uuid4())


def test_disk_file_storage_save_and_get_workflow(tmp_path: Path) -> None:
    """Test saving and retrieving a Workflow in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    plan = Plan(
        plan_context=PlanContext(query="query", tool_list=[]),
        steps=[],
    )
    workflow = Workflow(plan_id=plan.id)
    storage.save_workflow(workflow)
    retrieved_workflow = storage.get_workflow(workflow.id)

    assert retrieved_workflow.id == workflow.id

    with pytest.raises(WorkflowNotFoundError):
        storage.get_workflow(uuid4())


def test_disk_file_storage_invalid_plan_retrieval(tmp_path: Path) -> None:
    """Test handling of invalid Plan data in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    invalid_file = tmp_path / "plan-invalid.json"
    invalid_file.write_text('{"id": "not-a-valid-uuid"}')  # Write invalid JSON

    with pytest.raises(PlanNotFoundError):
        storage.get_plan(uuid4())


def test_disk_file_storage_invalid_workflow_retrieval(tmp_path: Path) -> None:
    """Test handling of invalid Workflow data in DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    invalid_file = tmp_path / "workflow-invalid.json"
    invalid_file.write_text('{"id": "not-a-valid-uuid"}')  # Write invalid JSON

    with pytest.raises(WorkflowNotFoundError):
        storage.get_workflow(uuid4())
