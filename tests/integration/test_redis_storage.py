"""Integration tests for RedisStorage using testcontainers."""

from __future__ import annotations

import pytest
from testcontainers.redis import RedisContainer

from portia.end_user import EndUser
from portia.execution_agents.output import AgentMemoryValue, LocalDataValue
from portia.plan import Plan, PlanContext
from portia.plan_run import PlanRun, PlanRunState
from portia.storage import RedisStorage
from tests.utils import get_test_plan_run


@pytest.fixture(scope="module")
def redis_container():
    """Provide a Redis container for integration tests."""
    with RedisContainer("redis:7-alpine") as redis:
        yield redis


@pytest.fixture
def redis_storage(redis_container):
    """Provide a RedisStorage instance connected to the test container."""
    redis_url = redis_container.get_connection_url()
    storage = RedisStorage(redis_url=redis_url)
    # Clean up any existing data
    storage._sync_client.flushdb()
    return storage


def test_redis_storage_plan_lifecycle(redis_storage: RedisStorage) -> None:
    """Test the full lifecycle of plan storage in Redis."""
    plan, _ = get_test_plan_run()

    # Test that plan doesn't exist initially
    assert not redis_storage.plan_exists(plan.id)

    # Save the plan
    redis_storage.save_plan(plan)

    # Check that plan now exists
    assert redis_storage.plan_exists(plan.id)

    # Retrieve the plan
    retrieved_plan = redis_storage.get_plan(plan.id)
    assert retrieved_plan.id == plan.id
    assert retrieved_plan.plan_context.query == plan.plan_context.query
    assert retrieved_plan.steps == plan.steps


def test_redis_storage_plan_run_lifecycle(redis_storage: RedisStorage) -> None:
    """Test the full lifecycle of plan run storage in Redis."""
    plan, plan_run = get_test_plan_run()

    # Save the plan run
    redis_storage.save_plan_run(plan_run)

    # Retrieve the plan run
    retrieved_run = redis_storage.get_plan_run(plan_run.id)
    assert retrieved_run.id == plan_run.id
    assert retrieved_run.plan_id == plan_run.plan_id
    assert retrieved_run.state == plan_run.state

    # Update the plan run state
    plan_run.state = PlanRunState.COMPLETE
    redis_storage.save_plan_run(plan_run)

    # Retrieve and verify the updated state
    updated_run = redis_storage.get_plan_run(plan_run.id)
    assert updated_run.state == PlanRunState.COMPLETE


def test_redis_storage_get_plan_runs(redis_storage: RedisStorage) -> None:
    """Test retrieving multiple plan runs with filtering."""
    plan, plan_run1 = get_test_plan_run()
    plan_run2 = PlanRun(plan_id=plan.id, end_user_id="user2")
    plan_run2.state = PlanRunState.COMPLETE

    # Save multiple plan runs
    redis_storage.save_plan_run(plan_run1)
    redis_storage.save_plan_run(plan_run2)

    # Get all plan runs
    all_runs = redis_storage.get_plan_runs()
    assert len(all_runs.results) == 2
    assert all_runs.count == 2

    # Get only completed plan runs
    completed_runs = redis_storage.get_plan_runs(run_state=PlanRunState.COMPLETE)
    assert len(completed_runs.results) == 1
    assert completed_runs.results[0].id == plan_run2.id

    # Get only in-progress plan runs
    in_progress_runs = redis_storage.get_plan_runs(run_state=PlanRunState.IN_PROGRESS)
    assert len(in_progress_runs.results) == 1
    assert in_progress_runs.results[0].id == plan_run1.id


def test_redis_storage_output_lifecycle(redis_storage: RedisStorage) -> None:
    """Test storing and retrieving plan run outputs."""
    plan, plan_run = get_test_plan_run()
    output = LocalDataValue(value="test value", summary="test summary")

    # Save the output
    result = redis_storage.save_plan_run_output("test_output", output, plan_run.id)

    # Verify the result is an AgentMemoryValue
    assert isinstance(result, AgentMemoryValue)
    assert result.output_name == "test_output"
    assert result.plan_run_id == plan_run.id
    assert result.summary == "test summary"

    # Retrieve the output
    retrieved_output = redis_storage.get_plan_run_output("test_output", plan_run.id)
    assert retrieved_output.value == "test value"
    assert retrieved_output.summary == "test summary"


def test_redis_storage_end_user_lifecycle(redis_storage: RedisStorage) -> None:
    """Test storing and retrieving end users."""
    end_user = EndUser(
        external_id="test_user",
        name="Test User",
        additional_data={"favorite_color": "blue"},
    )

    # Test that user doesn't exist initially
    assert redis_storage.get_end_user("test_user") is None

    # Save the end user
    saved_user = redis_storage.save_end_user(end_user)
    assert saved_user.external_id == "test_user"
    assert saved_user.name == "Test User"

    # Retrieve the end user
    retrieved_user = redis_storage.get_end_user("test_user")
    assert retrieved_user is not None
    assert retrieved_user.external_id == "test_user"
    assert retrieved_user.name == "Test User"
    assert retrieved_user.get_additional_data("favorite_color") == "blue"

    # Update the end user with additional data
    end_user.additional_data["favorite_food"] = "pizza"
    redis_storage.save_end_user(end_user)

    # Verify that both fields are present
    updated_user = redis_storage.get_end_user("test_user")
    assert updated_user is not None
    assert updated_user.get_additional_data("favorite_color") == "blue"
    assert updated_user.get_additional_data("favorite_food") == "pizza"


def test_redis_storage_get_plan_by_query(redis_storage: RedisStorage) -> None:
    """Test retrieving plans by query."""
    plan1 = Plan(
        plan_context=PlanContext(query="test query 1", tool_ids=[]),
        steps=[],
    )
    plan2 = Plan(
        plan_context=PlanContext(query="test query 2", tool_ids=[]),
        steps=[],
    )

    # Save plans
    redis_storage.save_plan(plan1)
    redis_storage.save_plan(plan2)

    # Retrieve plan by query
    found_plan = redis_storage.get_plan_by_query("test query 1")
    assert found_plan.id == plan1.id
    assert found_plan.plan_context.query == "test query 1"

    # Test with non-existent query
    from portia.errors import StorageError

    with pytest.raises(StorageError, match="No plan found for query"):
        redis_storage.get_plan_by_query("nonexistent query")


@pytest.mark.asyncio
async def test_redis_storage_async_plan_operations(redis_storage: RedisStorage) -> None:
    """Test async plan operations."""
    plan, _ = get_test_plan_run()

    # Test async save and get
    await redis_storage.asave_plan(plan)
    assert await redis_storage.aplan_exists(plan.id)

    retrieved_plan = await redis_storage.aget_plan(plan.id)
    assert retrieved_plan.id == plan.id


@pytest.mark.asyncio
async def test_redis_storage_async_plan_run_operations(redis_storage: RedisStorage) -> None:
    """Test async plan run operations."""
    plan, plan_run = get_test_plan_run()

    # Test async save and get
    await redis_storage.asave_plan_run(plan_run)

    retrieved_run = await redis_storage.aget_plan_run(plan_run.id)
    assert retrieved_run.id == plan_run.id

    # Test async get_plan_runs
    all_runs = await redis_storage.aget_plan_runs()
    assert len(all_runs.results) >= 1


@pytest.mark.asyncio
async def test_redis_storage_async_output_operations(redis_storage: RedisStorage) -> None:
    """Test async output operations."""
    plan, plan_run = get_test_plan_run()
    output = LocalDataValue(value="async test value", summary="async test summary")

    # Test async save and get
    result = await redis_storage.asave_plan_run_output("async_output", output, plan_run.id)
    assert isinstance(result, AgentMemoryValue)

    retrieved_output = await redis_storage.aget_plan_run_output("async_output", plan_run.id)
    assert retrieved_output.value == "async test value"


@pytest.mark.asyncio
async def test_redis_storage_async_end_user_operations(redis_storage: RedisStorage) -> None:
    """Test async end user operations."""
    end_user = EndUser(
        external_id="async_user",
        name="Async User",
    )

    # Test async save and get
    await redis_storage.asave_end_user(end_user)

    retrieved_user = await redis_storage.aget_end_user("async_user")
    assert retrieved_user is not None
    assert retrieved_user.external_id == "async_user"
