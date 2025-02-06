"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from portia.errors import StorageError
from portia.plan import Plan, PlanContext, PlanUUID
from portia.storage import (
    AdditionalStorage,
    DiskFileStorage,
    InMemoryStorage,
    PlanStorage,
    PortiaCloudStorage,
    WorkflowListResponse,
    WorkflowStorage,
)
from portia.workflow import Workflow, WorkflowState, WorkflowUUID
from tests.utils import get_test_config, get_test_tool_call, get_test_workflow

if TYPE_CHECKING:
    from pathlib import Path

    from portia.tool_call import ToolCallRecord


def test_storage_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyStorage(WorkflowStorage, PlanStorage, AdditionalStorage):
        """Override to test base."""

        def save_plan(self, plan: Plan) -> None:
            return super().save_plan(plan)  # type: ignore  # noqa: PGH003

        def get_plan(self, plan_id: PlanUUID) -> Plan:
            return super().get_plan(plan_id)  # type: ignore  # noqa: PGH003

        def save_workflow(self, workflow: Workflow) -> None:
            return super().save_workflow(workflow)  # type: ignore  # noqa: PGH003

        def get_workflow(self, workflow_id: WorkflowUUID) -> Workflow:
            return super().get_workflow(workflow_id)  # type: ignore  # noqa: PGH003

        def get_workflows(
            self,
            workflow_state: WorkflowState | None = None,
            page: int | None = None,
        ) -> WorkflowListResponse:
            return super().get_workflows(workflow_state, page)  # type: ignore  # noqa: PGH003

        def save_tool_call(self, tool_call: ToolCallRecord) -> None:
            return super().save_tool_call(tool_call)  # type: ignore  # noqa: PGH003

    storage = MyStorage()
    plan = Plan(plan_context=PlanContext(query="", tool_ids=[]), steps=[])
    workflow = Workflow(
        plan_id=plan.id,
    )

    tool_call = get_test_tool_call(workflow)

    with pytest.raises(NotImplementedError):
        storage.save_plan(plan)

    with pytest.raises(NotImplementedError):
        storage.get_plan(plan.id)

    with pytest.raises(NotImplementedError):
        storage.save_workflow(workflow)

    with pytest.raises(NotImplementedError):
        storage.get_workflow(workflow.id)

    with pytest.raises(NotImplementedError):
        storage.get_workflows()

    with pytest.raises(NotImplementedError):
        storage.save_tool_call(tool_call)


def test_in_memory_storage() -> None:
    """Test in memory storage."""
    storage = InMemoryStorage()
    (plan, workflow) = get_test_workflow()
    storage.save_plan(plan)
    assert storage.get_plan(plan.id) == plan
    storage.save_workflow(workflow)
    assert storage.get_workflow(workflow.id) == workflow
    assert storage.get_workflows().results == [workflow]
    assert storage.get_workflows(WorkflowState.FAILED).results == []


def test_disk_storage(tmp_path: Path) -> None:
    """Test disk storage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    (plan, workflow) = get_test_workflow()
    storage.save_plan(plan)
    assert storage.get_plan(plan.id) == plan
    storage.save_workflow(workflow)
    assert storage.get_workflow(workflow.id) == workflow
    all_workflows = storage.get_workflows()
    assert all_workflows.results == [workflow]
    assert storage.get_workflows(WorkflowState.FAILED).results == []


def test_portia_cloud_storage() -> None:
    """Test PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    workflow = Workflow(
        id=WorkflowUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
    )
    tool_call = get_test_tool_call(workflow)

    # Simulate a failed response
    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.content = b"An error occurred."

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test save_plan failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_plan(plan)

        mock_post.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plans/",
            json={
                "id": str(plan.id),
                "steps": [],
                "query": plan.plan_context.query,
                "tool_ids": plan.plan_context.tool_ids,
            },
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_plan failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan(plan.id)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plans/{plan.id}/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.put", return_value=mock_response) as mock_put,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test save_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_workflow(workflow)

        mock_put.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/{workflow.id}/",
            json={
                "current_step_index": workflow.current_step_index,
                "state": workflow.state,
                "execution_context": workflow.execution_context.model_dump(mode="json"),
                "outputs": workflow.outputs.model_dump(mode="json"),
                "plan_id": str(workflow.plan_id),
            },
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_workflow(workflow.id)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/{workflow.id}/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_workflows(WorkflowState.READY_TO_RESUME)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/?workflow_state=READY_TO_RESUME",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_workflows()

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/?",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/tool-calls/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            json={
                "workflow_id": str(tool_call.workflow_id),
                "tool_name": tool_call.tool_name,
                "step": tool_call.step,
                "end_user_id": tool_call.end_user_id or "",
                "additional_data": tool_call.additional_data,
                "input": tool_call.input,
                "output": tool_call.output,
                "status": tool_call.status,
                "latency_seconds": tool_call.latency_seconds,
            },
            timeout=10,
        )


def test_portia_cloud_storage_errors() -> None:
    """Test PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    workflow = Workflow(
        id=WorkflowUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
    )

    tool_call = get_test_tool_call(workflow)
    with (
        patch("httpx.post", side_effect=TimeoutError()) as mock_post,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test save_plan failure
        with pytest.raises(StorageError):
            storage.save_plan(plan)

        mock_post.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plans/",
            json={
                "id": str(plan.id),
                "steps": [],
                "query": plan.plan_context.query,
                "tool_ids": plan.plan_context.tool_ids,
            },
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", side_effect=TimeoutError()) as mock_post,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test get_plan failure
        with pytest.raises(StorageError):
            storage.get_plan(plan.id)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plans/{plan.id}/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.put", side_effect=TimeoutError()) as mock_put,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test save_workflow failure
        with pytest.raises(StorageError):
            storage.save_workflow(workflow)

        mock_put.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/{workflow.id}/",
            json={
                "current_step_index": workflow.current_step_index,
                "state": workflow.state,
                "execution_context": workflow.execution_context.model_dump(mode="json"),
                "outputs": workflow.outputs.model_dump(mode="json"),
                "plan_id": str(workflow.plan_id),
            },
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", side_effect=TimeoutError()) as mock_post,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError):
            storage.get_workflow(workflow.id)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/{workflow.id}/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", side_effect=TimeoutError()) as mock_post,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError):
            storage.get_workflows()

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/?",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", side_effect=TimeoutError()) as mock_post,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError):
            storage.get_workflows(workflow_state=WorkflowState.COMPLETE, page=10)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/workflows/?page=10&workflow_state=COMPLETE",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            timeout=10,
        )

    with (
        patch("httpx.post", side_effect=TimeoutError()) as mock_post,
        patch("httpx.get", side_effect=TimeoutError()) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError):
            storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/tool-calls/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            json={
                "workflow_id": str(tool_call.workflow_id),
                "tool_name": tool_call.tool_name,
                "step": tool_call.step,
                "end_user_id": tool_call.end_user_id or "",
                "additional_data": tool_call.additional_data,
                "input": tool_call.input,
                "output": tool_call.output,
                "status": tool_call.status,
                "latency_seconds": tool_call.latency_seconds,
            },
            timeout=10,
        )
