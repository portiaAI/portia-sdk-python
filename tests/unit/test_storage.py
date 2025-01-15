"""Test simple agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from portia.errors import StorageError
from portia.plan import Plan, PlanContext
from portia.storage import PlanStorage, PortiaCloudStorage, WorkflowStorage
from portia.workflow import Workflow
from tests.utils import get_test_config


def test_storage_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyStorage(WorkflowStorage, PlanStorage):
        """Override to test base."""

        def save_plan(self, plan: Plan) -> None:
            return super().save_plan(plan)  # type: ignore  # noqa: PGH003

        def get_plan(self, plan_id: UUID) -> Plan:
            return super().get_plan(plan_id)  # type: ignore  # noqa: PGH003

        def save_workflow(self, workflow: Workflow) -> None:
            return super().save_workflow(workflow)  # type: ignore  # noqa: PGH003

        def get_workflow(self, workflow_id: UUID) -> Workflow:
            return super().get_workflow(workflow_id)  # type: ignore  # noqa: PGH003

    storage = MyStorage()
    plan = Plan(plan_context=PlanContext(query="", tool_ids=[]), steps=[])
    workflow = Workflow(
        plan_id=plan.id,
    )
    with pytest.raises(NotImplementedError):
        storage.save_plan(plan)

    with pytest.raises(NotImplementedError):
        storage.get_plan(plan.id)

    with pytest.raises(NotImplementedError):
        storage.save_workflow(workflow)

    with pytest.raises(NotImplementedError):
        storage.get_workflow(workflow.id)


def test_portia_cloud_storage() -> None:
    """Test PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    workflow = Workflow(
        id=UUID("87654321-4321-8765-4321-876543218765"),
        plan_id=plan.id,
    )

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
            url="https://api.porita.dev/api/v0/plans/",
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
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_plan failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan(plan.id)

        mock_get.assert_called_once_with(
            url=f"https://api.porita.dev/api/v0/plans/{plan.id}/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test save_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_workflow(workflow)

        mock_post.assert_called_once_with(
            url="https://api.porita.dev/api/v0/workflows/",
            json={
                "id": str(workflow.id),
                "current_step_index": workflow.current_step_index,
                "state": workflow.state,
                "execution_context": workflow.execution_context,
                "outputs": workflow.outputs,
                "plan_id": str(workflow.plan_id),
            },
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
        )

    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
        patch("httpx.get", return_value=mock_response) as mock_get,
    ):
        # Test get_workflow failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_workflow(workflow.id)

        mock_get.assert_called_once_with(
            url=f"https://api.porita.dev/api/v0/workflows/{workflow.id}/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
        )
