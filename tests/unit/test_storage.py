"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from portia.errors import StorageError
from portia.plan import Plan, PlanContext, PlanUUID
from portia.plan_run import PlanRun, PlanRunState, PlanRunUUID
from portia.storage import (
    AdditionalStorage,
    DiskFileStorage,
    InMemoryStorage,
    PlanRunListResponse,
    PlanStorage,
    PortiaCloudStorage,
    RunStorage,
)
from tests.utils import get_test_config, get_test_plan_run, get_test_tool_call

if TYPE_CHECKING:
    from pathlib import Path

    from portia.tool_call import ToolCallRecord


def test_storage_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyStorage(RunStorage, PlanStorage, AdditionalStorage):
        """Override to test base."""

        def save_plan(self, plan: Plan) -> None:
            return super().save_plan(plan)  # type: ignore  # noqa: PGH003

        def get_plan(self, plan_id: PlanUUID) -> Plan:
            return super().get_plan(plan_id)  # type: ignore  # noqa: PGH003

        def save_plan_run(self, plan_run: PlanRun) -> None:
            return super().save_plan_run(plan_run)  # type: ignore  # noqa: PGH003

        def get_plan_run(self, plan_run_id: PlanRunUUID) -> PlanRun:
            return super().get_plan_run(plan_run_id)  # type: ignore  # noqa: PGH003

        def get_plan_runs(
            self,
            run_state: PlanRunState | None = None,
            page: int | None = None,
        ) -> PlanRunListResponse:
            return super().get_plan_runs(run_state, page)  # type: ignore  # noqa: PGH003

        def save_tool_call(self, tool_call: ToolCallRecord) -> None:
            return super().save_tool_call(tool_call)  # type: ignore  # noqa: PGH003

    storage = MyStorage()
    plan = Plan(plan_context=PlanContext(query="", tool_ids=[]), steps=[])
    plan_run = PlanRun(
        plan_id=plan.id,
    )

    tool_call = get_test_tool_call(plan_run)

    with pytest.raises(NotImplementedError):
        storage.save_plan(plan)

    with pytest.raises(NotImplementedError):
        storage.get_plan(plan.id)

    with pytest.raises(NotImplementedError):
        storage.save_plan_run(plan_run)

    with pytest.raises(NotImplementedError):
        storage.get_plan_run(plan_run.id)

    with pytest.raises(NotImplementedError):
        storage.get_plan_runs()

    with pytest.raises(NotImplementedError):
        storage.save_tool_call(tool_call)


def test_in_memory_storage() -> None:
    """Test in memory storage."""
    storage = InMemoryStorage()
    (plan, plan_run) = get_test_plan_run()
    storage.save_plan(plan)
    assert storage.get_plan(plan.id) == plan
    storage.save_plan_run(plan_run)
    assert storage.get_plan_run(plan_run.id) == plan_run
    assert storage.get_plan_runs().results == [plan_run]
    assert storage.get_plan_runs(PlanRunState.FAILED).results == []


def test_disk_storage(tmp_path: Path) -> None:
    """Test disk storage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    (plan, plan_run) = get_test_plan_run()
    storage.save_plan(plan)
    assert storage.get_plan(plan.id) == plan
    storage.save_plan_run(plan_run)
    assert storage.get_plan_run(plan_run.id) == plan_run
    all_runs = storage.get_plan_runs()
    assert all_runs.results == [plan_run]
    assert storage.get_plan_runs(PlanRunState.FAILED).results == []


def test_portia_cloud_storage() -> None:
    """Test PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
    )
    tool_call = get_test_tool_call(plan_run)

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
        # Test save_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_plan_run(plan_run)

        mock_put.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
            json={
                "current_step_index": plan_run.current_step_index,
                "state": plan_run.state,
                "execution_context": plan_run.execution_context.model_dump(mode="json"),
                "outputs": plan_run.outputs.model_dump(mode="json"),
                "plan_id": str(plan_run.plan_id),
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
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan_run(plan_run.id)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
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
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan_runs(PlanRunState.READY_TO_RESUME)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?run_state=READY_TO_RESUME",
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
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan_runs()

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?",
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
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/tool-calls/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            json={
                "plan_run_id": str(tool_call.plan_run_id),
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
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
    )

    tool_call = get_test_tool_call(plan_run)
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
        # Test save_run failure
        with pytest.raises(StorageError):
            storage.save_plan_run(plan_run)

        mock_put.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
            json={
                "current_step_index": plan_run.current_step_index,
                "state": plan_run.state,
                "execution_context": plan_run.execution_context.model_dump(mode="json"),
                "outputs": plan_run.outputs.model_dump(mode="json"),
                "plan_id": str(plan_run.plan_id),
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
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.get_plan_run(plan_run.id)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/{plan_run.id}/",
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
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.get_plan_runs()

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?",
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
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.get_plan_runs(run_state=PlanRunState.COMPLETE, page=10)

        mock_get.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/plan-runs/?page=10&run_state=COMPLETE",
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
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url=f"{config.portia_api_endpoint}/api/v0/tool-calls/",
            headers={
                "Authorization": "Api-Key test_api_key",
                "Content-Type": "application/json",
            },
            json={
                "plan_run_id": str(tool_call.plan_run_id),
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
