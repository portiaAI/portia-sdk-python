"""Tests for async storage methods."""

import asyncio
import tempfile

import pytest

from portia.end_user import EndUser
from portia.errors import PlanNotFoundError, PlanRunNotFoundError
from portia.plan import Plan, PlanContext, PlanUUID
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunUUID
from portia.storage import DiskFileStorage, InMemoryStorage
from portia.tool_call import ToolCallRecord, ToolCallStatus


class TestAsyncStorageMethods:
    """Test async storage methods."""

    def test_async_plan_storage_methods(self) -> None:
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

    def test_async_run_storage_methods(self) -> None:
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

    def test_async_additional_storage_methods(self) -> None:
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

    def test_async_agent_memory_methods(self) -> None:
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

    def test_async_disk_file_storage_methods(self) -> None:
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

    def test_async_error_handling(self) -> None:
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

    def test_async_concurrent_operations(self) -> None:
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

    def test_async_methods_inheritance(self) -> None:
        """Test that async methods are properly inherited by concrete classes."""
        # Test that InMemoryStorage has all async methods
        storage = InMemoryStorage()

        # Check that all async methods exist
        assert hasattr(storage, "asave_plan")
        assert hasattr(storage, "aget_plan")
        assert hasattr(storage, "aget_plan_by_query")
        assert hasattr(storage, "aplan_exists")
        assert hasattr(storage, "aget_similar_plans")
        assert hasattr(storage, "asave_plan_run")
        assert hasattr(storage, "aget_plan_run")
        assert hasattr(storage, "aget_plan_runs")
        assert hasattr(storage, "asave_tool_call")
        assert hasattr(storage, "asave_end_user")
        assert hasattr(storage, "aget_end_user")
        assert hasattr(storage, "asave_plan_run_output")
        assert hasattr(storage, "aget_plan_run_output")

        # Test that DiskFileStorage has all async methods
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_storage = DiskFileStorage(temp_dir)

            # Check that all async methods exist
            assert hasattr(disk_storage, "asave_plan")
            assert hasattr(disk_storage, "aget_plan")
            assert hasattr(disk_storage, "aget_plan_by_query")
            assert hasattr(disk_storage, "aplan_exists")
            assert hasattr(disk_storage, "aget_similar_plans")
            assert hasattr(disk_storage, "asave_plan_run")
            assert hasattr(disk_storage, "aget_plan_run")
            assert hasattr(disk_storage, "aget_plan_runs")
            assert hasattr(disk_storage, "asave_tool_call")
            assert hasattr(disk_storage, "asave_end_user")
            assert hasattr(disk_storage, "aget_end_user")
            assert hasattr(disk_storage, "asave_plan_run_output")
            assert hasattr(disk_storage, "aget_plan_run_output")
