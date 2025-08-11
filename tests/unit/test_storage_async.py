"""Tests for async storage methods."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID

import httpx
import pytest
from pytest_httpx import HTTPXMock

from portia.end_user import EndUser
from portia.errors import PlanNotFoundError, PlanRunNotFoundError, StorageError
from portia.execution_agents.output import (
    AgentMemoryValue,
    LocalDataValue,
)
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunUUID
from portia.storage import (
    MAX_STORAGE_OBJECT_BYTES,
    DiskFileStorage,
    InMemoryStorage,
    PortiaCloudStorage,
)
from portia.tool_call import ToolCallRecord, ToolCallStatus
from tests.utils import get_test_config, get_test_tool_call


def test_async_plan_storage_methods() -> None:
    """Test async plan storage methods."""
    storage = InMemoryStorage()
    plan = Plan(
        id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
        plan_context=PlanContext(
            query="test query",
            tool_ids=["test_tool"],
        ),
        steps=[],
        plan_inputs=[],
    )

    # Test async save_plan
    async def test_save_plan() -> None:
        await storage.asave_plan(plan)
        assert storage.plan_exists(plan.id)

    asyncio.run(test_save_plan())

    # Test async get_plan
    async def test_get_plan() -> None:
        retrieved_plan = await storage.aget_plan(plan.id)
        assert retrieved_plan.id == plan.id
        assert retrieved_plan.plan_context.query == plan.plan_context.query

    asyncio.run(test_get_plan())

    # Test async plan_exists
    async def test_plan_exists() -> None:
        exists = await storage.aplan_exists(plan.id)
        assert exists is True

    asyncio.run(test_plan_exists())

    # Test async get_plan_by_query
    async def test_get_plan_by_query() -> None:
        retrieved_plan = await storage.aget_plan_by_query("test query")
        assert retrieved_plan.id == plan.id

    asyncio.run(test_get_plan_by_query())


def test_async_run_storage_methods() -> None:
    """Test async run storage methods."""
    storage = InMemoryStorage()
    plan_run = PlanRun(
        id=PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
        plan_id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
        end_user_id="test_user",
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
        outputs=PlanRunOutputs(),
        plan_run_inputs={},
    )

    # Test async save_plan_run
    async def test_save_plan_run() -> None:
        await storage.asave_plan_run(plan_run)
        retrieved_run = storage.get_plan_run(plan_run.id)
        assert retrieved_run.id == plan_run.id

    asyncio.run(test_save_plan_run())

    # Test async get_plan_run
    async def test_get_plan_run() -> None:
        retrieved_run = await storage.aget_plan_run(plan_run.id)
        assert retrieved_run.id == plan_run.id

    asyncio.run(test_get_plan_run())

    # Test async get_plan_runs
    async def test_get_plan_runs() -> None:
        runs_response = await storage.aget_plan_runs()
        assert len(runs_response.results) == 1
        assert runs_response.results[0].id == plan_run.id

    asyncio.run(test_get_plan_runs())


def test_async_additional_storage_methods() -> None:
    """Test async additional storage methods."""
    storage = InMemoryStorage()
    tool_call = ToolCallRecord(
        plan_run_id=PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
        tool_name="test_tool",
        step=1,
        end_user_id="test_user",
        input={"test": "input"},
        output="test output",
        status=ToolCallStatus.SUCCESS,
        latency_seconds=1.0,
    )

    # Test async save_tool_call
    async def test_save_tool_call() -> None:
        await storage.asave_tool_call(tool_call)
        # Tool calls are just logged, so we just verify no exception is raised

    asyncio.run(test_save_tool_call())

    # Test async save_end_user
    end_user = EndUser(
        external_id="test_user",
        name="Test User",
        email="test@example.com",
        phone_number="",
        additional_data={},
    )

    async def test_save_end_user() -> None:
        saved_user = await storage.asave_end_user(end_user)
        assert saved_user.external_id == end_user.external_id

    asyncio.run(test_save_end_user())

    # Test async get_end_user
    async def test_get_end_user() -> None:
        retrieved_user = await storage.aget_end_user("test_user")
        assert retrieved_user is not None
        assert retrieved_user.external_id == "test_user"

    asyncio.run(test_get_end_user())


def test_async_agent_memory_methods() -> None:
    """Test async agent memory methods."""
    storage = InMemoryStorage()
    from portia.execution_agents.output import LocalDataValue

    output = LocalDataValue(
        summary="test summary",
        value="test value",
    )

    # Test async save_plan_run_output
    async def test_save_plan_run_output() -> None:
        result = await storage.asave_plan_run_output(
            "test_output",
            output,
            PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
        )
        assert result.summary == "test summary"

    asyncio.run(test_save_plan_run_output())

    # Test async get_plan_run_output
    async def test_get_plan_run_output() -> None:
        retrieved_output = await storage.aget_plan_run_output(
            "test_output",
            PlanRunUUID.from_string("prun-87654321-4321-8765-4321-876543210987"),
        )
        assert retrieved_output.value == "test value"

    asyncio.run(test_get_plan_run_output())


def test_async_disk_file_storage_methods() -> None:
    """Test async disk file storage methods."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = DiskFileStorage(temp_dir)
        plan = Plan(
            id=PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678"),
            plan_context=PlanContext(
                query="test query",
                tool_ids=["test_tool"],
            ),
            steps=[],
            plan_inputs=[],
        )

        # Test async save_plan
        async def test_save_plan() -> None:
            await storage.asave_plan(plan)
            assert storage.plan_exists(plan.id)

        asyncio.run(test_save_plan())

        # Test async get_plan
        async def test_get_plan() -> None:
            retrieved_plan = await storage.aget_plan(plan.id)
            assert retrieved_plan.id == plan.id
            assert retrieved_plan.plan_context.query == plan.plan_context.query

        asyncio.run(test_get_plan())

        # Test async plan_exists
        async def test_plan_exists() -> None:
            exists = await storage.aplan_exists(plan.id)
            assert exists is True

        asyncio.run(test_plan_exists())


def test_async_error_handling() -> None:
    """Test that async methods properly propagate errors."""
    storage = InMemoryStorage()

    # Test that non-existent plan raises error
    async def test_get_nonexistent_plan() -> None:
        with pytest.raises(PlanNotFoundError):
            await storage.aget_plan(
                PlanUUID.from_string("plan-99999999-9999-9999-9999-999999999999")
            )

    asyncio.run(test_get_nonexistent_plan())

    # Test that non-existent plan_run raises error
    async def test_get_nonexistent_plan_run() -> None:
        with pytest.raises(PlanRunNotFoundError):
            await storage.aget_plan_run(
                PlanRunUUID.from_string("prun-99999999-9999-9999-9999-999999999999")
            )

    asyncio.run(test_get_nonexistent_plan_run())


def test_async_concurrent_operations() -> None:
    """Test that async methods can be called concurrently."""
    storage = InMemoryStorage()
    plans = []

    # Create multiple plans
    for i in range(5):
        plan = Plan(
            id=PlanUUID.from_string(f"plan-{i:08d}-1234-5678-1234-567812345678"),
            plan_context=PlanContext(
                query=f"test query {i}",
                tool_ids=["test_tool"],
            ),
            steps=[],
            plan_inputs=[],
        )
        plans.append(plan)

    # Test concurrent save operations
    async def test_concurrent_saves() -> None:
        tasks = [storage.asave_plan(plan) for plan in plans]
        await asyncio.gather(*tasks)

        # Verify all plans were saved
        for plan in plans:
            assert storage.plan_exists(plan.id)

    asyncio.run(test_concurrent_saves())

    # Test concurrent get operations
    async def test_concurrent_gets() -> None:
        tasks = [storage.aget_plan(plan.id) for plan in plans]
        retrieved_plans = await asyncio.gather(*tasks)

        # Verify all plans were retrieved correctly
        for i, retrieved_plan in enumerate(retrieved_plans):
            assert retrieved_plan.id == plans[i].id
            assert retrieved_plan.plan_context.query == plans[i].plan_context.query

    asyncio.run(test_concurrent_gets())


def test_async_portia_cloud_storage() -> None:
    """Test async PortiaCloudStorage raises StorageError on failure responses."""
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

    async def test_async_save_plan() -> None:
        with (
            patch.object(storage.async_client, "post", return_value=mock_response) as mock_post,
        ):
            # Test async save_plan failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.asave_plan(plan)

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

    asyncio.run(test_async_save_plan())

    async def test_async_get_plan() -> None:
        with (
            patch.object(storage.async_client, "get", return_value=mock_response) as mock_get,
        ):
            # Test async get_plan failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.aget_plan(plan.id)

            mock_get.assert_called_once_with(
                url=f"/api/v0/plans/{plan.id}/",
            )

    asyncio.run(test_async_get_plan())

    async def test_async_save_plan_run() -> None:
        with (
            patch.object(storage.async_client, "put", return_value=mock_response) as mock_put,
        ):
            # Test async save_run failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.asave_plan_run(plan_run)

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

    asyncio.run(test_async_save_plan_run())

    async def test_async_get_plan_run() -> None:
        with (
            patch.object(storage.async_client, "get", return_value=mock_response) as mock_get,
        ):
            # Test async get_run failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.aget_plan_run(plan_run.id)

            mock_get.assert_called_once_with(
                url=f"/api/v0/plan-runs/{plan_run.id}/",
            )

    asyncio.run(test_async_get_plan_run())

    async def test_async_get_plan_runs() -> None:
        with (
            patch.object(storage.async_client, "get", return_value=mock_response) as mock_get,
        ):
            # Test async get_runs failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.aget_plan_runs()

            mock_get.assert_called_once_with(
                url="/api/v0/plan-runs/?",
            )

    asyncio.run(test_async_get_plan_runs())

    async def test_async_save_tool_call() -> None:
        with (
            patch.object(storage.async_client, "post", return_value=mock_response) as mock_post,
        ):
            # Test async save_tool_call - should not raise an exception
            await storage.asave_tool_call(tool_call)

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

    asyncio.run(test_async_save_tool_call())

    async def test_async_get_end_user() -> None:
        with (
            patch.object(storage.async_client, "get", return_value=mock_response) as mock_get,
        ):
            # Test async get_end_user failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.aget_end_user(end_user.external_id)

            mock_get.assert_called_once_with(
                url=f"/api/v0/end-user/{end_user.external_id}/",
            )

    asyncio.run(test_async_get_end_user())

    async def test_async_save_end_user() -> None:
        with (
            patch.object(storage.async_client, "put", return_value=mock_response) as mock_put,
        ):
            # Test async save_end_user failure
            with pytest.raises(StorageError, match="An error occurred."):
                await storage.asave_end_user(end_user)

            mock_put.assert_called_once_with(
                url=f"/api/v0/end-user/{end_user.external_id}/",
                json=end_user.model_dump(mode="json"),
            )

    asyncio.run(test_async_save_end_user())


def test_async_portia_cloud_storage_errors() -> None:
    """Test async PortiaCloudStorage raises StorageError on failure responses."""
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

    async def test_async_save_plan_error() -> None:
        with (
            patch.object(storage.async_client, "post", side_effect=mock_exception) as mock_post,
        ):
            # Test async save_plan failure
            with pytest.raises(StorageError):
                await storage.asave_plan(plan)

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

    asyncio.run(test_async_save_plan_error())

    async def test_async_get_plan_error() -> None:
        with (
            patch.object(storage.async_client, "get", side_effect=mock_exception) as mock_get,
        ):
            # Test async get_plan failure
            with pytest.raises(StorageError):
                await storage.aget_plan(plan.id)

            mock_get.assert_called_once_with(
                url=f"/api/v0/plans/{plan.id}/",
            )

    asyncio.run(test_async_get_plan_error())

    async def test_async_save_plan_run_error() -> None:
        with (
            patch.object(storage.async_client, "put", side_effect=mock_exception) as mock_put,
        ):
            # Test async save_run failure
            with pytest.raises(StorageError):
                await storage.asave_plan_run(plan_run)

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

    asyncio.run(test_async_save_plan_run_error())

    async def test_async_get_plan_run_error() -> None:
        with (
            patch.object(storage.async_client, "get", side_effect=mock_exception) as mock_get,
        ):
            # Test async get_run failure
            with pytest.raises(StorageError):
                await storage.aget_plan_run(plan_run.id)

            mock_get.assert_called_once_with(
                url=f"/api/v0/plan-runs/{plan_run.id}/",
            )

    asyncio.run(test_async_get_plan_run_error())

    async def test_async_get_plan_runs_error() -> None:
        with (
            patch.object(storage.async_client, "get", side_effect=mock_exception) as mock_get,
        ):
            # Test async get_runs failure
            with pytest.raises(StorageError):
                await storage.aget_plan_runs()

            mock_get.assert_called_once_with(
                url="/api/v0/plan-runs/?",
            )

    asyncio.run(test_async_get_plan_runs_error())

    async def test_async_get_plan_runs_with_params_error() -> None:
        with (
            patch.object(storage.async_client, "get", side_effect=mock_exception) as mock_get,
        ):
            # Test async get_runs with parameters failure
            with pytest.raises(StorageError):
                await storage.aget_plan_runs(run_state=PlanRunState.COMPLETE, page=10)

            mock_get.assert_called_once_with(
                url="/api/v0/plan-runs/?page=10&run_state=COMPLETE",
            )

    asyncio.run(test_async_get_plan_runs_with_params_error())

    async def test_async_save_tool_call_error() -> None:
        with (
            patch.object(storage.async_client, "post", side_effect=mock_exception) as mock_post,
        ):
            # Test async save_tool_call - should not raise an exception
            await storage.asave_tool_call(tool_call)

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

    asyncio.run(test_async_save_tool_call_error())

    async def test_async_save_end_user_error() -> None:
        with (
            patch.object(storage.async_client, "put", side_effect=mock_exception) as mock_put,
        ):
            # Test async save_end_user failure
            with pytest.raises(StorageError):
                await storage.asave_end_user(end_user)

            mock_put.assert_called_once_with(
                url=f"/api/v0/end-user/{end_user.external_id}/",
                json=end_user.model_dump(mode="json"),
            )

    asyncio.run(test_async_save_end_user_error())

    async def test_async_get_end_user_error() -> None:
        with (
            patch.object(storage.async_client, "get", side_effect=mock_exception) as mock_get,
        ):
            # Test async get_end_user failure
            with pytest.raises(StorageError):
                await storage.aget_end_user(end_user.external_id)

            mock_get.assert_called_once_with(
                url=f"/api/v0/end-user/{end_user.external_id}/",
            )

    asyncio.run(test_async_get_end_user_error())


def test_async_portia_cloud_agent_memory(httpx_mock: HTTPXMock) -> None:
    """Test async PortiaCloudStorage agent memory."""
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

    async def test_async_save_plan_run_output() -> None:
        # Test saving an output
        httpx_mock.add_response(
            method="PUT",
            url=f"https://api.portialabs.ai/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
            status_code=200,
        )

        result = await agent_memory.asave_plan_run_output("test_output", output, plan_run.id)

        # Verify the PUT request was made correctly
        assert len(httpx_mock.get_requests()) == 1
        put_request = httpx_mock.get_requests()[0]
        assert put_request.method == "PUT"
        assert (
            put_request.url.path
            == f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/"
        )

        # Verify the result
        assert isinstance(result, AgentMemoryValue)
        assert result.output_name == "test_output"
        assert result.plan_run_id == plan_run.id
        assert result.summary == output.get_summary()
        assert Path(f".portia/cache/agent_memory/{plan_run.id}/test_output.json").is_file()

    asyncio.run(test_async_save_plan_run_output())

    async def test_async_get_plan_run_output_cached() -> None:
        # Test getting an output when it is cached locally
        with patch.object(agent_memory.async_client, "get") as mock_get:
            result = await agent_memory.aget_plan_run_output("test_output", plan_run.id)

            # Verify that we didn't call Portia Cloud because we have a cached value
            mock_get.assert_not_called()

            # Verify the returned output
            assert result.get_summary() == output.get_summary()
            assert result.get_value() == output.get_value()

    asyncio.run(test_async_get_plan_run_output_cached())

    async def test_async_get_plan_run_output_not_cached() -> None:
        # Test getting an output when it is not cached locally
        # Mock the metadata response
        httpx_mock.add_response(
            method="GET",
            url=f"https://api.portialabs.ai/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output2/",
            status_code=200,
            json={
                "summary": "test summary 2",
                "url": "https://example.com/output2",
            },
        )

        # Mock the value response
        httpx_mock.add_response(
            method="GET",
            url="https://example.com/output2",
            status_code=200,
            content=b"test value 2",
        )

        with (
            patch.object(agent_memory, "_read_from_cache", side_effect=FileNotFoundError),
            patch.object(agent_memory, "_write_to_cache") as mock_write_cache,
        ):
            result = await agent_memory.aget_plan_run_output("test_output2", plan_run.id)

            # Verify that both HTTP requests were made
            assert len(httpx_mock.get_requests()) >= 3  # Previous requests + 2 from get

            # Verify the metadata request
            metadata_request = httpx_mock.get_requests()[-2]
            assert metadata_request.method == "GET"
            assert (
                metadata_request.url.path
                == f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output2/"
            )

            # Verify the value request
            value_request = httpx_mock.get_requests()[-1]
            assert value_request.method == "GET"
            assert value_request.url == "https://example.com/output2"

            # Verify that it wrote to the local cache
            mock_write_cache.assert_called_once()

            # Verify the returned output
            assert result.get_summary() == "test summary 2"
            assert result.get_value() == "test value 2"

    asyncio.run(test_async_get_plan_run_output_not_cached())


def test_async_portia_cloud_agent_memory_errors() -> None:
    """Test async PortiaCloudStorage raises StorageError on agent memory failure responses."""
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

    async def test_async_save_plan_run_output_error() -> None:
        with (
            patch.object(
                agent_memory.async_form_client, "put", side_effect=mock_exception
            ) as mock_put,
        ):
            with pytest.raises(StorageError):
                await agent_memory.asave_plan_run_output("test_output", output, plan_run.id)

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

    asyncio.run(test_async_save_plan_run_output_error())

    async def test_async_get_plan_run_output_error() -> None:
        with (
            patch.object(
                agent_memory,
                "_read_from_cache",
                side_effect=FileNotFoundError,
            ) as mock_read_cache,
            patch.object(
                agent_memory.async_client, "get", side_effect=mock_exception
            ) as mock_get,
        ):
            with pytest.raises(StorageError):
                await agent_memory.aget_plan_run_output("test_output", plan_run.id)

            mock_read_cache.assert_called_once_with(
                f"{plan_run.id}/test_output.json", LocalDataValue
            )
            mock_get.assert_called_once_with(
                url=f"/api/v0/agent-memory/plan-runs/{plan_run.id}/outputs/test_output/",
            )

    asyncio.run(test_async_get_plan_run_output_error())

    async def test_async_save_plan_run_output_too_large() -> None:
        # Check with an output that's too large
        with (
            patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
            pytest.raises(StorageError),
        ):
            await agent_memory.asave_plan_run_output(
                "large_output",
                LocalDataValue(value="large value"),
                plan_run.id,
            )

    asyncio.run(test_async_save_plan_run_output_too_large())

    async def test_async_save_plan_run_output_request_too_large() -> None:
        # Test for 413 REQUEST_ENTITY_TOO_LARGE response status
        mock_response = MagicMock()
        mock_response.status_code = httpx.codes.REQUEST_ENTITY_TOO_LARGE
        mock_response.request = MagicMock()
        mock_response.request.content = b"Some content that's too large"

        with (
            patch.object(agent_memory.async_form_client, "put", return_value=mock_response),
            pytest.raises(StorageError),
        ):
            await agent_memory.asave_plan_run_output(
                "too_large_output",
                LocalDataValue(value="too large value"),
                plan_run.id,
            )

    asyncio.run(test_async_save_plan_run_output_request_too_large())

    async def test_async_save_plan_run_output_over_size_limit() -> None:
        # Test for response.request.content > MAX_STORAGE_OBJECT_BYTES
        mock_response = MagicMock()
        mock_response.status_code = httpx.codes.OK
        mock_response.request = MagicMock()
        mock_response.request.content = b"Some large content"

        with (
            patch.object(agent_memory.async_form_client, "put", return_value=mock_response),
            patch("sys.getsizeof", return_value=MAX_STORAGE_OBJECT_BYTES + 1),
            pytest.raises(StorageError),
        ):
            await agent_memory.asave_plan_run_output(
                "over_size_limit",
                LocalDataValue(value="value that creates a large request"),
                plan_run.id,
            )

    asyncio.run(test_async_save_plan_run_output_over_size_limit())


def test_async_similar_plans(httpx_mock: HTTPXMock) -> None:
    """Test the async similar_plans method."""
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

    async def test_async_get_similar_plans() -> None:
        plans = await storage.aget_similar_plans("Test query")
        assert len(plans) == 2
        assert plans[0].id == PlanUUID.from_string(mock_id)
        assert plans[1].id == PlanUUID.from_string(mock_id)

    asyncio.run(test_async_get_similar_plans())


def test_async_similar_plans_error(httpx_mock: HTTPXMock) -> None:
    """Test the async similar_plans method with an error."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)
    endpoint = config.portia_api_endpoint
    url = f"{endpoint}/api/v0/plans/embeddings/search/"
    httpx_mock.add_response(
        url=url,
        status_code=500,
    )

    async def test_async_get_similar_plans_error() -> None:
        with pytest.raises(StorageError):
            await storage.aget_similar_plans("Test query")

    asyncio.run(test_async_get_similar_plans_error())


def test_async_plan_exists_portia_cloud_storage() -> None:
    """Test async plan_exists method with PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[],
    )

    async def test_async_plan_exists_success() -> None:
        # Test when plan exists
        mock_success_response = MagicMock()
        mock_success_response.is_success = True
        with patch.object(
            storage.async_client, "get", return_value=mock_success_response
        ) as mock_get:
            exists = await storage.aplan_exists(plan.id)
            assert exists is True
            mock_get.assert_called_once_with(url=f"/api/v0/plans/{plan.id}/")

    asyncio.run(test_async_plan_exists_success())

    async def test_async_plan_exists_not_found() -> None:
        # Test when plan doesn't exist
        mock_failure_response = MagicMock()
        mock_failure_response.is_success = False
        with patch.object(
            storage.async_client, "get", return_value=mock_failure_response
        ) as mock_get:
            different_plan_id = PlanUUID()
            exists = await storage.aplan_exists(different_plan_id)
            assert exists is False
            mock_get.assert_called_once_with(url=f"/api/v0/plans/{different_plan_id}/")

    asyncio.run(test_async_plan_exists_not_found())

    async def test_async_plan_exists_api_error() -> None:
        # Test when API call fails
        with patch.object(
            storage.async_client, "get", side_effect=Exception("API Error")
        ) as mock_get:
            exists = await storage.aplan_exists(plan.id)
            assert exists is False
            mock_get.assert_called_once_with(url=f"/api/v0/plans/{plan.id}/")

    asyncio.run(test_async_plan_exists_api_error())


def test_async_get_plan_by_query_portia_cloud_storage(httpx_mock: HTTPXMock) -> None:
    """Test async get_plan_by_query method with PortiaCloudStorage."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    async def test_async_get_plan_by_query_success() -> None:
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
        found_plan = await storage.aget_plan_by_query("test query")
        assert found_plan.plan_context.query == "test query"
        assert found_plan.id == PlanUUID.from_string(
            "plan-00000000-0000-0000-0000-000000000000"
        )

    asyncio.run(test_async_get_plan_by_query_success())

    async def test_async_get_plan_by_query_not_found() -> None:
        # Test with no matching plans
        endpoint = config.portia_api_endpoint
        url = f"{endpoint}/api/v0/plans/embeddings/search/"
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
            await storage.aget_plan_by_query("non-existent query")

    asyncio.run(test_async_get_plan_by_query_not_found())


def test_async_get_plan_by_query_portia_cloud_storage_error(httpx_mock: HTTPXMock) -> None:
    """Test async get_plan_by_query method with PortiaCloudStorage when API fails."""
    config = get_test_config(portia_api_key="test_api_key")
    storage = PortiaCloudStorage(config)

    async def test_async_get_plan_by_query_api_error() -> None:
        endpoint = config.portia_api_endpoint
        url = f"{endpoint}/api/v0/plans/embeddings/search/"

        # Test with API error
        httpx_mock.add_response(
            url=url,
            status_code=500,
            method="POST",
        )

        with pytest.raises(StorageError):
            await storage.aget_plan_by_query("test query")

    asyncio.run(test_async_get_plan_by_query_api_error())

    async def test_async_get_plan_by_query_network_error() -> None:
        endpoint = config.portia_api_endpoint
        url = f"{endpoint}/api/v0/plans/embeddings/search/"

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
            await storage.aget_plan_by_query("test query")

    asyncio.run(test_async_get_plan_by_query_network_error())
