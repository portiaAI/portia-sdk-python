"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from portia.errors import StorageError
from portia.execution_agents.output import (
    FileMemoryValue,
    LocalMemoryValue,
    Output,
    RemoteMemoryValue,
)
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

    mock_response = MagicMock()
    mock_response.is_success = False
    mock_response.content = b"An error occurred."

    with (
        patch.object(storage.client, "post", return_value=mock_response) as mock_post,
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        # Test save_plan failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_plan(plan)

        mock_post.assert_called_once_with(
            url="/api/v0/plans/",
            json={
                "id": str(plan.id),
                "steps": [],
                "query": plan.plan_context.query,
                "tool_ids": plan.plan_context.tool_ids,
            },
        )

    with (
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        # Test get_plan failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan(plan.id)

        mock_get.assert_called_once_with(
            url=f"/api/v0/plans/{plan.id}/",
        )

    with (
        patch.object(storage.client, "put", return_value=mock_response) as mock_put,
    ):
        # Test save_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_plan_run(plan_run)

        mock_put.assert_called_once_with(
            url=f"/api/v0/plan-runs/{plan_run.id}/",
            json={
                "current_step_index": plan_run.current_step_index,
                "state": plan_run.state,
                "execution_context": plan_run.execution_context.model_dump(mode="json"),
                "outputs": plan_run.outputs.model_dump(mode="json"),
                "plan_id": str(plan_run.plan_id),
            },
        )

    with (
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan_run(plan_run.id)

        mock_get.assert_called_once_with(
            url=f"/api/v0/plan-runs/{plan_run.id}/",
        )

    with (
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan_runs()

        mock_get.assert_called_once_with(
            url="/api/v0/plan-runs/?",
        )

    with (
        patch.object(storage.client, "post", return_value=mock_response) as mock_post,
    ):
        # Test save_tool_call failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url="/api/v0/tool-calls/",
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

    mock_exception = RuntimeError("An error occurred.")
    with (
        patch.object(storage.client, "post", side_effect=mock_exception) as mock_post,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test save_plan failure
        with pytest.raises(StorageError):
            storage.save_plan(plan)

        mock_post.assert_called_once_with(
            url="/api/v0/plans/",
            json={
                "id": str(plan.id),
                "steps": [],
                "query": plan.plan_context.query,
                "tool_ids": plan.plan_context.tool_ids,
            },
        )
    with (
        patch.object(storage.client, "post", side_effect=mock_exception) as mock_post,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test get_plan failure
        with pytest.raises(StorageError):
            storage.get_plan(plan.id)

        mock_get.assert_called_once_with(
            url=f"/api/v0/plans/{plan.id}/",
        )

    with (
        patch.object(storage.client, "put", side_effect=mock_exception) as mock_put,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test save_run failure
        with pytest.raises(StorageError):
            storage.save_plan_run(plan_run)

        mock_put.assert_called_once_with(
            url=f"/api/v0/plan-runs/{plan_run.id}/",
            json={
                "current_step_index": plan_run.current_step_index,
                "state": plan_run.state,
                "execution_context": plan_run.execution_context.model_dump(mode="json"),
                "outputs": plan_run.outputs.model_dump(mode="json"),
                "plan_id": str(plan_run.plan_id),
            },
        )

    with (
        patch.object(storage.client, "post", side_effect=mock_exception) as mock_post,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.get_plan_run(plan_run.id)

        mock_get.assert_called_once_with(
            url=f"/api/v0/plan-runs/{plan_run.id}/",
        )

    with (
        patch.object(storage.client, "post", side_effect=mock_exception) as mock_post,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.get_plan_runs()

        mock_get.assert_called_once_with(
            url="/api/v0/plan-runs/?",
        )

    with (
        patch.object(storage.client, "post", side_effect=mock_exception) as mock_post,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.get_plan_runs(run_state=PlanRunState.COMPLETE, page=10)

        mock_get.assert_called_once_with(
            url="/api/v0/plan-runs/?page=10&run_state=COMPLETE",
        )

    with (
        patch.object(storage.client, "post", side_effect=mock_exception) as mock_post,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError):
            storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url="/api/v0/tool-calls/",
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
        )


def test_in_memory_storage_outputs() -> None:
    """Test saving and retrieving outputs in InMemoryStorage."""
    storage = InMemoryStorage()
    plan_run = get_test_plan_run()[1]
    output = Output(value="test output", summary="test summary")
    output_name = "test_output"

    # Test saving output
    stored_output = storage.save_plan_run_output(output_name, output, plan_run.id)
    assert stored_output.summary == output.summary
    assert stored_output.value == LocalMemoryValue(value=output.value)

    # Test retrieving output
    retrieved_output = storage.get_plan_run_output(output_name, plan_run.id)
    assert retrieved_output.value == LocalMemoryValue(value=output.value)
    assert retrieved_output.summary == output.summary
    assert retrieved_output.value_for_prompt() == output.summary
    assert retrieved_output.full_value() == output.value

    # Test retrieving non-existent output
    with pytest.raises(KeyError):
        storage.get_plan_run_output("non_existent", plan_run.id)


def test_disk_file_storage_outputs(tmp_path: Path) -> None:
    """Test saving and retrieving outputs in DiskFileStorage."""
    storage = DiskFileStorage(str(tmp_path))
    plan_run = get_test_plan_run()[1]
    output = Output(value="test output", summary="test summary")
    output_name = "test_output"

    # Test saving output
    stored_output = storage.save_plan_run_output(output_name, output, plan_run.id)
    assert stored_output.summary == output.summary
    assert isinstance(stored_output.value, FileMemoryValue)
    assert stored_output.value.path == str(tmp_path / str(plan_run.id) / f"{output_name}.json")

    # Test retrieving output
    retrieved_output = storage.get_plan_run_output(output_name, plan_run.id)
    assert retrieved_output.summary == output.summary
    assert isinstance(retrieved_output.value, FileMemoryValue)
    assert retrieved_output.value.path == str(tmp_path / str(plan_run.id) / f"{output_name}.json")
    assert retrieved_output.value_for_prompt() == output.summary
    assert retrieved_output.full_value() == output.value

    # Test retrieving non-existent output
    with pytest.raises(FileNotFoundError):
        storage.get_plan_run_output("non_existent", plan_run.id)


def test_portia_cloud_storage_outputs() -> None:
    """Test saving and retrieving outputs in PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)
    plan_run = get_test_plan_run()[1]
    output = Output(value="test output", summary="test summary")
    output_name = "test_output"

    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json = MagicMock(
        return_value={
            "name": output_name,
            "summary": output.summary,
            "url": "https://example.com/output",
        },
    )

    with (
        patch.object(storage.client, "put", return_value=mock_response) as mock_put,
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        # Test saving output
        stored_output = storage.save_plan_run_output(output_name, output, plan_run.id)
        mock_put.assert_called_once_with(
            url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/{output_name}/",
            json={
                "value": output.value.encode("utf-8"),
                "summary": output.summary,
            },
        )
        assert stored_output.value == RemoteMemoryValue(url="https://example.com/output")
        assert stored_output.summary == output.summary

        # Test retrieving output
        retrieved_output = storage.get_plan_run_output(output_name, plan_run.id)
        assert stored_output.value == RemoteMemoryValue(url="https://example.com/output")
        assert retrieved_output.summary == output.summary
        assert stored_output.value_for_prompt() == output.summary
        mock_get.assert_called_once_with(
            url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/{output_name}/",
        )

    # Test retrieving full value from remote URL
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.text = "test-value"
    with (
        patch("httpx.Client.get", return_value=mock_http_response) as mock_http_get,
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        retrieved_output = storage.get_plan_run_output(output_name, plan_run.id)
        assert retrieved_output.full_value() == "test-value"
        mock_http_get.assert_called_once_with("https://example.com/output")

    # Test error handling
    mock_error_response = MagicMock()
    mock_error_response.is_success = False
    mock_error_response.content = b"An error occurred."

    with patch.object(storage.client, "put", return_value=mock_error_response) as mock_put:
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_plan_run_output(output_name, output, plan_run.id)
        mock_put.assert_called_once()

    with patch.object(storage.client, "get", return_value=mock_error_response) as mock_get:
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_plan_run_output(output_name, plan_run.id)
        mock_get.assert_called_once()
