"""Test simple agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID

import httpx
import pytest

from portia.end_user import EndUser
from portia.errors import StorageError
from portia.execution_agents.output import (
    AgentMemoryValue,
    LocalDataValue,
)
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID
from portia.plan_run import PlanRun, PlanRunState, PlanRunUUID
from portia.storage import (
    MAX_STORAGE_OBJECT_BYTES,
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
    from pytest_httpx import HTTPXMock

    from portia.tool_call import ToolCallRecord


def test_storage_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyStorage(RunStorage, PlanStorage, AdditionalStorage):
        """Override to test base."""

        def save_plan(self, plan: Plan) -> None:
            return super().save_plan(plan)  # type: ignore  # noqa: PGH003

        def get_plan(self, plan_id: PlanUUID) -> Plan:
            return super().get_plan(plan_id)  # type: ignore  # noqa: PGH003

        def get_plan_by_query(self, query: str) -> Plan:
            return super().get_plan_by_query(query)  # type: ignore  # noqa: PGH003

        def plan_exists(self, plan_id: PlanUUID) -> bool:
            return super().plan_exists(plan_id)  # type: ignore  # noqa: PGH003

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

        def save_end_user(self, end_user: EndUser) -> EndUser:
            return super().save_end_user(end_user)  # type: ignore  # noqa: PGH003

        def get_end_user(self, external_id: str) -> EndUser:
            return super().get_end_user(external_id)  # type: ignore  # noqa: PGH003

    storage = MyStorage()
    plan = Plan(plan_context=PlanContext(query="", tool_ids=[]), steps=[])
    plan_run = PlanRun(
        plan_id=plan.id,
        end_user_id="test123",
    )

    tool_call = get_test_tool_call(plan_run)

    end_user = EndUser(external_id="123")

    with pytest.raises(NotImplementedError):
        storage.save_plan(plan)

    with pytest.raises(NotImplementedError):
        storage.get_plan(plan.id)

    with pytest.raises(NotImplementedError):
        storage.get_plan_by_query("test query")

    with pytest.raises(NotImplementedError):
        storage.plan_exists(plan.id)

    with pytest.raises(NotImplementedError):
        storage.save_plan_run(plan_run)

    with pytest.raises(NotImplementedError):
        storage.get_plan_run(plan_run.id)

    with pytest.raises(NotImplementedError):
        storage.get_plan_runs()

    with pytest.raises(NotImplementedError):
        storage.save_tool_call(tool_call)

    with pytest.raises(NotImplementedError):
        storage.save_end_user(end_user)

    with pytest.raises(NotImplementedError):
        storage.get_end_user(end_user.external_id)


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
    saved_output_1 = storage.save_plan_run_output(
        "test name",
        LocalDataValue(value="test value"),
        plan_run.id,
    )
    assert storage.get_plan_run_output("test name", plan_run.id) == LocalDataValue(
        value="test value"
    )
    # Check that we ignore outputs that are already in agent memory
    saved_output_2 = storage.save_plan_run_output(
        "test name",
        saved_output_1,
        plan_run.id,
    )
    assert saved_output_2 == saved_output_1
    # Check with an output that's too large
    with (
        patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
        pytest.raises(StorageError),
    ):
        storage.save_plan_run_output(
            "large_output",
            LocalDataValue(value="large value"),
            plan_run.id,
        )

    # This just logs, but check it doesn't cause any issues
    tool_call = get_test_tool_call(plan_run)
    storage.save_tool_call(tool_call)
    # Check with a very large output too
    tool_call.output = "a" * 100000
    storage.save_tool_call(tool_call)

    end_user = EndUser(
        external_id="123",
        additional_data={"favorite_sport": "football"},
    )

    assert storage.get_end_user("123") is None
    assert storage.save_end_user(end_user) == end_user
    user = storage.get_end_user("123")
    assert user is not None
    assert user.get_additional_data("favorite_sport") == "football"
    end_user.additional_data["day"] = "monday"
    assert storage.save_end_user(end_user) == end_user
    user = storage.get_end_user("123")
    assert user is not None
    assert user.get_additional_data("favorite_sport") == "football"
    assert user.get_additional_data("day") == "monday"


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
    storage.save_plan_run_output("test name", LocalDataValue(value="test value"), plan_run.id)
    assert storage.get_plan_run_output("test name", plan_run.id) == LocalDataValue(
        value="test value"
    )

    # Check with an output that's too large
    with (
        patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
        pytest.raises(StorageError),
    ):
        storage.save_plan_run_output(
            "large_output",
            LocalDataValue(value="large value"),
            plan_run.id,
        )

    # This just logs, but check it doesn't cause any issues
    tool_call = get_test_tool_call(plan_run)
    storage.save_tool_call(tool_call)
    # Check with a very large output too
    tool_call.output = "a" * 100000
    storage.save_tool_call(tool_call)

    end_user = EndUser(
        external_id="123",
        additional_data={"favorite_sport": "football"},
    )

    assert storage.get_end_user("123") is None
    assert storage.save_end_user(end_user) == end_user
    user = storage.get_end_user("123")
    assert user is not None
    assert user.get_additional_data("favorite_sport") == "football"
    end_user.additional_data["day"] = "monday"
    assert storage.save_end_user(end_user) == end_user
    user = storage.get_end_user("123")
    assert user is not None
    assert user.get_additional_data("favorite_sport") == "football"
    assert user.get_additional_data("day") == "monday"


def test_portia_cloud_storage() -> None:
    """Test PortiaCloudStorage raises StorageError on failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
        plan_inputs=[
            PlanInput(name="key1", description="Test input 1"),
            PlanInput(name="key2", description="Test input 2"),
        ],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
        plan_run_inputs={
            "param1": LocalDataValue(value="test"),
            "param2": LocalDataValue(value=456),
        },
    )
    tool_call = get_test_tool_call(plan_run)

    end_user = EndUser(external_id="123")

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
                "plan_inputs": [
                    {"name": "key1", "description": "Test input 1", "value": None},
                    {"name": "key2", "description": "Test input 2", "value": None},
                ],
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
                "end_user": plan_run.end_user_id,
                "outputs": plan_run.outputs.model_dump(mode="json"),
                "plan_id": str(plan_run.plan_id),
                "plan_run_inputs": {
                    "param1": {
                        "value": "test",
                        "summary": None,
                    },
                    "param2": {
                        "value": "456",
                        "summary": None,
                    },
                },
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
        # Test save_tool_call - should not raise an exception
        storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url="/api/v0/tool-calls/",
            json={
                "plan_run_id": str(tool_call.plan_run_id),
                "tool_name": tool_call.tool_name,
                "step": tool_call.step,
                "end_user_id": tool_call.end_user_id or "",
                "input": tool_call.input,
                "output": tool_call.output,
                "status": tool_call.status,
                "latency_seconds": tool_call.latency_seconds,
            },
        )

    with (
        patch.object(storage.client, "get", return_value=mock_response) as mock_get,
    ):
        # Test get_run failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.get_end_user(end_user.external_id)

        mock_get.assert_called_once_with(
            url=f"/api/v0/end-user/{end_user.external_id}/",
        )

    with (
        patch.object(storage.client, "put", return_value=mock_response) as mock_put,
    ):
        # Test save_tool_call failure
        with pytest.raises(StorageError, match="An error occurred."):
            storage.save_end_user(end_user)

        mock_put.assert_called_once_with(
            url=f"/api/v0/end-user/{end_user.external_id}/",
            json=end_user.model_dump(mode="json"),
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
        end_user_id="test123",
    )

    tool_call = get_test_tool_call(plan_run)

    end_user = EndUser(external_id="123")

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
                "plan_inputs": [],
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
                "end_user": plan_run.end_user_id,
                "outputs": plan_run.outputs.model_dump(mode="json"),
                "plan_id": str(plan_run.plan_id),
                "plan_run_inputs": plan_run.plan_run_inputs,
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
    ):
        # Test save_tool_call - should not raise an exception
        storage.save_tool_call(tool_call)

        mock_post.assert_called_once_with(
            url="/api/v0/tool-calls/",
            json={
                "plan_run_id": str(tool_call.plan_run_id),
                "tool_name": tool_call.tool_name,
                "step": tool_call.step,
                "end_user_id": tool_call.end_user_id or "",
                "input": tool_call.input,
                "output": tool_call.output,
                "status": tool_call.status,
                "latency_seconds": tool_call.latency_seconds,
            },
        )

    with (
        patch.object(storage.client, "put", side_effect=mock_exception) as mock_put,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test save_end_user failure
        with pytest.raises(StorageError):
            storage.save_end_user(end_user)

        mock_put.assert_called_once_with(
            url=f"/api/v0/end-user/{end_user.external_id}/",
            json=end_user.model_dump(mode="json"),
        )

    with (
        patch.object(storage.client, "put", side_effect=mock_exception) as mock_put,
        patch.object(storage.client, "get", side_effect=mock_exception) as mock_get,
    ):
        # Test get_end_user failure
        with pytest.raises(StorageError):
            storage.get_end_user(end_user.external_id)

        mock_get.assert_called_once_with(
            url=f"/api/v0/end-user/{end_user.external_id}/",
        )


def test_portia_cloud_agent_memory(httpx_mock: HTTPXMock) -> None:
    """Test PortiaCloudStorage agent memory."""
    config = get_test_config(portia_api_key="test_api_key")
    agent_memory = PortiaCloudStorage(config)
    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
    )
    output = LocalDataValue(value="test value", summary="test summary")
    mock_success_response = MagicMock()
    mock_success_response.is_success = True

    # Test saving an output
    with (
        patch.object(
            agent_memory.form_client,
            "put",
            return_value=mock_success_response,
        ) as mock_put,
    ):
        result = agent_memory.save_plan_run_output("test_output", output, plan_run.id)

        mock_put.assert_called_once_with(
            url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
            files={
                "value": (
                    "output",
                    ANY,
                ),
            },
            data={
                "summary": output.get_summary(),
            },
        )
        assert isinstance(result, AgentMemoryValue)
        assert result.output_name == "test_output"
        assert result.plan_run_id == plan_run.id
        assert result.summary == output.get_summary()
        assert Path(f".portia/cache/agent_memory/{plan_run.id}/test_output.json").is_file()

    # Test getting an output when it is cached locally
    with (
        patch.object(agent_memory.client, "get") as mock_get,
    ):
        result = agent_memory.get_plan_run_output("test_output", plan_run.id)

        # Verify that we didn't call Portia Cloud because we have a cached value
        mock_get.assert_not_called()

        # Verify the returned output
        assert result.get_summary() == output.get_summary()
        assert result.get_value() == output.get_value()

    # Test getting an output when it is not cached locally
    mock_output_response = MagicMock()
    mock_output_response.is_success = True
    mock_output_response.json.return_value = {
        "summary": "test summary",
        "url": "https://example.com/output",
    }
    httpx_mock.add_response(
        url="https://example.com/output",
        status_code=200,
        content=b"test value",
    )

    with (
        patch.object(agent_memory, "_read_from_cache", side_effect=FileNotFoundError),
        patch.object(agent_memory.client, "get", return_value=mock_output_response) as mock_get,
        patch.object(agent_memory, "_write_to_cache") as mock_write_cache,
    ):
        result = agent_memory.get_plan_run_output("test_output", plan_run.id)

        # Verify that it fetched from Portia Cloud
        mock_get.assert_called_once_with(
            url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
        )

        # Verify that it fetched the value from the URL using the httpx client
        assert len(httpx_mock.get_requests()) == 1

        # Verify that it wrote to the local cache
        mock_write_cache.assert_called_once()

        # Verify the returned output
        assert result.get_summary() == "test summary"
        assert result.get_value() == "test value"


def test_portia_cloud_agent_memory_local_cache_expiry() -> None:
    """Test PortiaCloudStorage agent memory."""
    config = get_test_config(portia_api_key="test_api_key")
    agent_memory = PortiaCloudStorage(config)
    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
    )
    output = LocalDataValue(value="test value", summary="test summary")
    mock_success_response = MagicMock()
    mock_success_response.is_success = True

    mock_success_response = MagicMock()
    mock_success_response.is_success = True

    with (
        patch.object(
            agent_memory.form_client,
            "put",
            return_value=mock_success_response,
        ),
        patch.object(agent_memory.client, "get"),
    ):
        # Write 21 outputs to the cache (cache size is 20)
        for i in range(21):
            agent_memory.save_plan_run_output(f"test_output_{i}", output, plan_run.id)

        # Check that the cache only stores 20 entries
        cache_files = list(Path(agent_memory.cache_dir).glob("**/*.json"))
        assert len(cache_files) == 20
        assert "test_output_20.json" in [file.name for file in cache_files]


def test_portia_cloud_agent_memory_errors() -> None:
    """Test PortiaCloudStorage raises StorageError on agent memory failure responses."""
    config = get_test_config(portia_api_key="test_api_key")
    agent_memory = PortiaCloudStorage(config)
    plan = Plan(
        id=PlanUUID(uuid=UUID("12345678-1234-5678-1234-567812345678")),
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    plan_run = PlanRun(
        id=PlanRunUUID(uuid=UUID("87654321-4321-8765-4321-876543218765")),
        plan_id=plan.id,
        end_user_id="test123",
    )
    output = LocalDataValue(value="test value", summary="test summary")

    mock_exception = RuntimeError("An error occurred.")
    with (
        patch.object(agent_memory.form_client, "put", side_effect=mock_exception) as mock_put,
    ):
        with pytest.raises(StorageError):
            agent_memory.save_plan_run_output("test_output", output, plan_run.id)

        mock_put.assert_called_once_with(
            url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
            files={
                "value": (
                    "output",
                    ANY,
                ),
            },
            data={
                "summary": output.get_summary(),
            },
        )

    with (
        patch.object(
            agent_memory,
            "_read_from_cache",
            side_effect=FileNotFoundError,
        ) as mock_read_cache,
        patch.object(agent_memory.client, "get", side_effect=mock_exception) as mock_get,
    ):
        with pytest.raises(StorageError):
            agent_memory.get_plan_run_output("test_output", plan_run.id)

        mock_read_cache.assert_called_once_with(f"{plan_run.id}/test_output.json", LocalDataValue)
        mock_get.assert_called_once_with(
            url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
        )

    # Check with an output that's too large
    with (
        patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
        pytest.raises(StorageError),
    ):
        agent_memory.save_plan_run_output(
            "large_output",
            LocalDataValue(value="large value"),
            plan_run.id,
        )

    # Test for 413 REQUEST_ENTITY_TOO_LARGE response status
    mock_response = MagicMock()
    mock_response.status_code = httpx.codes.REQUEST_ENTITY_TOO_LARGE
    mock_response.request = MagicMock()
    mock_response.request.content = b"Some content that's too large"

    with (
        patch.object(agent_memory.form_client, "put", return_value=mock_response),
        pytest.raises(StorageError),
    ):
        agent_memory.save_plan_run_output(
            "too_large_output",
            LocalDataValue(value="too large value"),
            plan_run.id,
        )

    # Test for response.request.content > MAX_STORAGE_OBJECT_BYTES
    mock_response = MagicMock()
    mock_response.status_code = httpx.codes.OK
    mock_response.request = MagicMock()
    mock_response.request.content = b"Some large content"

    with (
        patch.object(agent_memory.form_client, "put", return_value=mock_response),
        patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
        pytest.raises(StorageError),
    ):
        agent_memory.save_plan_run_output(
            "over_size_limit",
            LocalDataValue(value="value that creates a large request"),
            plan_run.id,
        )


def test_similar_plans(httpx_mock: HTTPXMock) -> None:
    """Test the similar_plans method."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)
    mock_id = "plan-00000000-0000-0000-0000-000000000000"
    mock_response = {
        "id": mock_id,
        "steps": [],
        "query": "Test query",
        "tool_ids": [],
    }
    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=200,
        method="POST",
        match_json={
            "query": "Test query",
            "threshold": 0.5,
            "limit": 5,
        },
        json=[mock_response, mock_response],
    )
    plans = storage.get_similar_plans("Test query")
    assert len(plans) == 2
    assert plans[0].id == PlanUUID.from_string(mock_id)
    assert plans[1].id == PlanUUID.from_string(mock_id)


def test_similar_plans_error(httpx_mock: HTTPXMock) -> None:
    """Test the similar_plans method with an error."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)
    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=500,
    )
    with pytest.raises(StorageError):
        storage.get_similar_plans("Test query")


def test_plan_exists_in_memory_storage() -> None:
    """Test plan_exists method with InMemoryStorage."""
    storage = InMemoryStorage()
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[],
    )

    # Test non-existent plan
    assert not storage.plan_exists(plan.id)

    # Save plan and test again
    storage.save_plan(plan)
    assert storage.plan_exists(plan.id)

    # Test with different plan ID
    different_plan_id = PlanUUID()
    assert not storage.plan_exists(different_plan_id)


def test_plan_exists_disk_storage(tmp_path: Path) -> None:
    """Test plan_exists method with DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[],
    )

    # Test non-existent plan
    assert not storage.plan_exists(plan.id)

    # Save plan and test again
    storage.save_plan(plan)
    assert storage.plan_exists(plan.id)

    # Test with different plan ID
    different_plan_id = PlanUUID()
    assert not storage.plan_exists(different_plan_id)


def test_plan_exists_portia_cloud_storage() -> None:
    """Test plan_exists method with PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[],
    )

    # Test when plan exists
    mock_success_response = MagicMock()
    mock_success_response.is_success = True
    with patch.object(storage.client, "get", return_value=mock_success_response) as mock_get:
        assert storage.plan_exists(plan.id)
        mock_get.assert_called_once_with(url=f"/api/v0/plans/{plan.id}/")

    # Test when plan doesn't exist
    mock_failure_response = MagicMock()
    mock_failure_response.is_success = False
    with patch.object(storage.client, "get", return_value=mock_failure_response) as mock_get:
        different_plan_id = PlanUUID()
        assert not storage.plan_exists(different_plan_id)
        mock_get.assert_called_once_with(url=f"/api/v0/plans/{different_plan_id}/")

    # Test when API call fails
    with patch.object(storage.client, "get", side_effect=Exception("API Error")) as mock_get:
        assert not storage.plan_exists(plan.id)
        mock_get.assert_called_once_with(url=f"/api/v0/plans/{plan.id}/")


def test_get_plan_by_query_in_memory_storage() -> None:
    """Test get_plan_by_query method with InMemoryStorage."""
    storage = InMemoryStorage()

    # Create test plans with different queries
    plan1 = Plan(
        plan_context=PlanContext(query="test query 1", tool_ids=[]),
        steps=[],
    )
    plan2 = Plan(
        plan_context=PlanContext(query="test query 2", tool_ids=[]),
        steps=[],
    )
    plan3 = Plan(
        plan_context=PlanContext(
            query="test query 1", tool_ids=["tool1"]
        ),  # Same query, different tool_ids
        steps=[],
    )

    # Save plans
    storage.save_plan(plan1)
    storage.save_plan(plan2)
    storage.save_plan(plan3)

    # Test finding existing plan
    found_plan = storage.get_plan_by_query("test query 1")
    assert found_plan.plan_context.query == "test query 1"
    # Should return the first one found (plan1 in this case)
    assert found_plan.id == plan1.id

    # Test finding another existing plan
    found_plan = storage.get_plan_by_query("test query 2")
    assert found_plan.plan_context.query == "test query 2"
    assert found_plan.id == plan2.id

    # Test finding non-existent plan
    with pytest.raises(StorageError, match="No plan found for query: non-existent query"):
        storage.get_plan_by_query("non-existent query")

    # Test with empty storage
    empty_storage = InMemoryStorage()
    with pytest.raises(StorageError, match="No plan found for query: any query"):
        empty_storage.get_plan_by_query("any query")


def test_get_plan_by_query_disk_storage(tmp_path: Path) -> None:
    """Test get_plan_by_query method with DiskFileStorage."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))

    # Create test plans with different queries
    plan1 = Plan(
        plan_context=PlanContext(query="test query 1", tool_ids=[]),
        steps=[],
    )
    plan2 = Plan(
        plan_context=PlanContext(query="test query 2", tool_ids=[]),
        steps=[],
    )
    plan3 = Plan(
        plan_context=PlanContext(
            query="test query 1", tool_ids=["tool1"]
        ),  # Same query, different tool_ids
        steps=[],
    )

    # Save plans
    storage.save_plan(plan1)
    storage.save_plan(plan2)
    storage.save_plan(plan3)

    # Test finding existing plan
    found_plan = storage.get_plan_by_query("test query 1")
    assert found_plan.plan_context.query == "test query 1"
    # Should return the first one found
    assert found_plan.id in [plan3.id, plan1.id]

    # Test finding another existing plan
    found_plan = storage.get_plan_by_query("test query 2")
    assert found_plan.plan_context.query == "test query 2"
    assert found_plan.id == plan2.id

    # Test finding non-existent plan
    with pytest.raises(StorageError, match="No plan found for query: non-existent query"):
        storage.get_plan_by_query("non-existent query")

    # Test with empty storage directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()  # Create the directory first
    empty_storage = DiskFileStorage(storage_dir=str(empty_dir))
    with pytest.raises(StorageError, match="No plan found for query: any query"):
        empty_storage.get_plan_by_query("any query")

    # Test with directory containing non-plan files
    mixed_storage = DiskFileStorage(storage_dir=str(tmp_path / "mixed"))
    mixed_storage._ensure_storage()

    # Create a non-plan file
    non_plan_file = tmp_path / "mixed" / "not_a_plan.txt"
    non_plan_file.write_text("This is not a plan file")

    with pytest.raises(StorageError, match="No plan found for query: any query"):
        mixed_storage.get_plan_by_query("any query")


def test_get_plan_by_query_disk_storage_multiple_plans(tmp_path: Path) -> None:
    """Test that get_plan_by_query returns the first plan that matches the query."""
    storage = DiskFileStorage(storage_dir=str(tmp_path))

    # Create test plans with the same query
    plan1 = Plan(
        plan_context=PlanContext(query="different query", tool_ids=[]),
        steps=[],
    )
    plan2 = Plan(
        plan_context=PlanContext(query="different query", tool_ids=["tool1"]),
        steps=[],
    )
    plan3 = Plan(
        plan_context=PlanContext(query="same query", tool_ids=["tool1", "tool2"]),
        steps=[],
    )

    # Save plans in order (plan1 first, then plan2, then plan3)
    storage.save_plan(plan1)
    storage.save_plan(plan2)
    storage.save_plan(plan3)

    # Should return the first plan that matches the query, which might vary depending on the id
    # This is not guaranteed to be the most recent plan. Unfortunately getting the file creation
    # time is not supported on all platforms.
    found_plan = storage.get_plan_by_query("same query")
    assert found_plan.plan_context.query == "same query"
    assert found_plan.id == plan3.id


def test_get_plan_by_query_portia_cloud_storage(httpx_mock: HTTPXMock) -> None:
    """Test get_plan_by_query method with PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    # Mock the get_similar_plans response
    mock_plan_response = {
        "id": "plan-00000000-0000-0000-0000-000000000000",
        "steps": [],
        "query": "test query",
        "tool_ids": ["tool1"],
    }

    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=200,
        method="POST",
        match_json={
            "query": "test query",
            "threshold": 1.0,
            "limit": 1,
        },
        json=[mock_plan_response],
    )

    # Test finding existing plan
    found_plan = storage.get_plan_by_query("test query")
    assert found_plan.plan_context.query == "test query"
    assert found_plan.id == PlanUUID.from_string("plan-00000000-0000-0000-0000-000000000000")

    # Test with no matching plans
    httpx_mock.add_response(
        url=url,
        status_code=200,
        method="POST",
        match_json={
            "query": "non-existent query",
            "threshold": 1.0,
            "limit": 1,
        },
        json=[],
    )

    with pytest.raises(StorageError, match="No plan found for query: non-existent query"):
        storage.get_plan_by_query("non-existent query")


def test_get_plan_by_query_portia_cloud_storage_error(httpx_mock: HTTPXMock) -> None:
    """Test get_plan_by_query method with PortiaCloudStorage when API fails."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"

    # Test with API error
    httpx_mock.add_response(
        url=url,
        status_code=500,
        method="POST",
    )

    with pytest.raises(StorageError):
        storage.get_plan_by_query("test query")

    # Test with network error - we need to use a callback to raise the exception
    def raise_connection_error(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    httpx_mock.add_callback(
        raise_connection_error,
        url=url,
        method="POST",
        match_json={
            "query": "test query",
            "threshold": 1.0,
            "limit": 1,
        },
    )

    with pytest.raises(StorageError):
        storage.get_plan_by_query("test query")


def test_get_plan_by_query_edge_cases() -> None:
    """Test get_plan_by_query with edge cases."""
    # Test with empty query
    storage = InMemoryStorage()
    plan = Plan(
        plan_context=PlanContext(query="", tool_ids=[]),
        steps=[],
    )
    storage.save_plan(plan)

    found_plan = storage.get_plan_by_query("")
    assert found_plan.plan_context.query == ""

    # Test with very long query
    long_query = "a" * 1000
    plan2 = Plan(
        plan_context=PlanContext(query=long_query, tool_ids=[]),
        steps=[],
    )
    storage.save_plan(plan2)

    found_plan = storage.get_plan_by_query(long_query)
    assert found_plan.plan_context.query == long_query

    # Test with special characters in query
    special_query = "test query with !@#$%^&*() characters"
    plan3 = Plan(
        plan_context=PlanContext(query=special_query, tool_ids=[]),
        steps=[],
    )
    storage.save_plan(plan3)

    found_plan = storage.get_plan_by_query(special_query)
    assert found_plan.plan_context.query == special_query
