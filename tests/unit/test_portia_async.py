"""Async tests for portia classes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, HttpUrl, SecretStr

from portia.clarification import (
    ActionClarification,
    Clarification,
    InputClarification,
    ValueConfirmationClarification,
)
from portia.config import Config, GenerativeModelsConfig, StorageClass
from portia.end_user import EndUser
from portia.errors import (
    InvalidPlanRunStateError,
    PlanError,
    PlanNotFoundError,
    PlanRunNotFoundError,
)
from portia.execution_agents.output import LocalDataValue
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.introspection_agents.introspection_agent import (
    COMPLETED_OUTPUT,
    SKIPPED_OUTPUT,
    PreStepIntrospection,
    PreStepIntrospectionOutcome,
)
from portia.plan import (
    Plan,
    PlanBuilder,
    PlanContext,
    PlanInput,
    PlanUUID,
    ReadOnlyPlan,
    ReadOnlyStep,
    Step,
    Variable,
)
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunUUID
from portia.planning_agents.base_planning_agent import StepsOrError
from portia.portia import Portia
from portia.storage import StorageError
from portia.telemetry.views import PortiaFunctionCallTelemetryEvent
from portia.tool import ReadyResponse, Tool, ToolRunContext, _ArgsSchemaPlaceholder
from portia.tool_registry import ToolRegistry
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    TestClarificationHandler,
    get_test_config,
    get_test_plan_run,
)


@pytest.mark.asyncio
async def test_portia_agenerate_plan(
    portia: Portia, planning_model: MagicMock, telemetry: MagicMock
) -> None:
    """Test async planning a query."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan = await portia.aplan(query)

    telemetry.capture.assert_called_once_with(
        PortiaFunctionCallTelemetryEvent(
            function_name="portia_aplan",
            function_call_details={
                "tools": None,
                "example_plans_provided": False,
                "end_user_provided": False,
                "plan_inputs_provided": False,
            },
            name="portia_function_call",
        )
    )

    assert plan.plan_context.query == query


@pytest.mark.asyncio
async def test_portia_agenerate_plan_error(
    portia: Portia, planning_model: MagicMock, telemetry: MagicMock
) -> None:
    """Test async planning a query that returns an error."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error="could not plan",
    )
    with pytest.raises(PlanError):
        await portia.aplan(query)

    # Check that the telemetry event was captured despite the error.
    telemetry.capture.assert_called_once_with(
        PortiaFunctionCallTelemetryEvent(
            function_name="portia_aplan",
            function_call_details={
                "tools": None,
                "example_plans_provided": False,
                "end_user_provided": False,
                "plan_inputs_provided": False,
            },
            name="portia_function_call",
        )
    )


@pytest.mark.asyncio
async def test_portia_agenerate_plan_with_tools(
    portia: Portia, planning_model: MagicMock, telemetry: MagicMock
) -> None:
    """Test async planning a query with tools."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan = await portia.aplan(query, tools=["add_tool"])

    telemetry.capture.assert_called_once_with(
        PortiaFunctionCallTelemetryEvent(
            function_name="portia_aplan",
            function_call_details={
                "tools": "add_tool",
                "example_plans_provided": False,
                "end_user_provided": False,
                "plan_inputs_provided": False,
            },
            name="portia_function_call",
        )
    )

    assert plan.plan_context.query == query
    assert plan.plan_context.tool_ids == ["add_tool"]


@pytest.mark.asyncio
async def test_portia_aplan_with_use_cached_plan_success(portia: Portia) -> None:
    """Test async planning with use_cached_plan=True when cached plan exists."""
    query = "example query"

    # Create a cached plan
    cached_plan = Plan(
        plan_context=PlanContext(query=query, tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(cached_plan)

    # Mock the storage.get_plan_by_query to return the cached plan
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", return_value=cached_plan
    ) as mock_get_cached:
        plan = await portia.aplan(query, use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify the cached plan was returned
        assert plan.id == cached_plan.id
        assert plan.plan_context.query == query


@pytest.mark.asyncio
async def test_portia_aplan_with_use_cached_plan_not_found(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async planning with use_cached_plan=True when no cached plan exists."""
    query = "example query"

    # Mock the storage.get_plan_by_query to raise StorageError
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", side_effect=StorageError("No plan found for query")
    ) as mock_get_cached:
        # Mock the planning model to return a successful plan
        planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

        plan = await portia.aplan(query, use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify a new plan was generated (not the cached one)
        assert plan.plan_context.query == query
        assert plan.id != "plan-00000000-0000-0000-0000-000000000000"  # Not a default UUID


@pytest.mark.asyncio
async def test_portia_aplan_with_use_cached_plan_false(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async planning with use_cached_plan=False (default behavior)."""
    query = "example query"

    # Create a cached plan
    cached_plan = Plan(
        plan_context=PlanContext(query=query, tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(cached_plan)

    # Mock the planning model to return a successful plan
    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

    # Mock the storage.get_plan_by_query to ensure it's not called
    with mock.patch.object(portia.storage, "aget_plan_by_query") as mock_get_cached:
        plan = await portia.aplan(query, use_cached_plan=False)

        # Verify get_plan_by_query was NOT called
        mock_get_cached.assert_not_called()

        # Verify a new plan was generated
        assert plan.plan_context.query == query
        assert plan.id != cached_plan.id  # Should be a different plan


@pytest.mark.asyncio
async def test_portia_aplan_with_use_cached_plan_and_tools(portia: Portia) -> None:
    """Test async planning with use_cached_plan=True and tools when cached plan exists."""
    query = "example query"

    # Create a cached plan with tools
    cached_plan = Plan(
        plan_context=PlanContext(query=query, tool_ids=["add_tool", "subtract_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(cached_plan)

    # Mock the storage.get_plan_by_query to return the cached plan
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", return_value=cached_plan
    ) as mock_get_cached:
        plan = await portia.aplan(query, tools=["add_tool"], use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify the cached plan was returned (tools parameter should be ignored when using
        # cached plan)
        assert plan.id == cached_plan.id
        assert plan.plan_context.query == query
        assert plan.plan_context.tool_ids == ["add_tool", "subtract_tool"]  # Original cached tools


@pytest.mark.asyncio
async def test_portia_aplan_with_use_cached_plan_storage_error_logging(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async planning with use_cached_plan=True when storage error occurs."""
    query = "example query"

    # Mock the storage.get_plan_by_query to raise StorageError
    with mock.patch.object(
        portia.storage, "get_plan_by_query", side_effect=StorageError("Storage error")
    ) as mock_get_cached:
        # Mock the planning model to return a successful plan
        planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

        plan = await portia.aplan(query, use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify a new plan was generated despite the storage error
        assert plan.plan_context.query == query
        assert plan.id != "plan-00000000-0000-0000-0000-000000000000"  # Not a default UUID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan_inputs",
    [
        [
            PlanInput(name="$num_a", description="Number A"),
            PlanInput(name="$num_b", description="Number B"),
        ],
        [
            {"name": "$num_a", "description": "Number A"},
            {"name": "$num_b"},
        ],
        ["$num_a", "$num_b"],
        [
            {"incorrect_key": "$num_a", "error": "Error"},
        ],
        "error",
    ],
)
async def test_portia_aplan_with_plan_inputs(
    portia: Portia,
    planning_model: MagicMock,
    plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str],
    telemetry: MagicMock,
) -> None:
    """Test async planning with various plan input formats."""
    query = "example query"

    # Mock the planning model to return a successful plan
    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

    if plan_inputs == "error":
        with pytest.raises(ValueError, match="Invalid plan inputs received"):
            await portia.aplan(query, plan_inputs=plan_inputs)
    elif (
        isinstance(plan_inputs, list)
        and plan_inputs
        and isinstance(plan_inputs[0], dict)
        and "incorrect_key" in plan_inputs[0]
    ):
        with pytest.raises(ValueError, match="Plan input must have a name and description"):
            await portia.aplan(query, plan_inputs=plan_inputs)
    else:
        plan = await portia.aplan(query, plan_inputs=plan_inputs)

        telemetry.capture.assert_called_once_with(
            PortiaFunctionCallTelemetryEvent(
                function_name="portia_aplan",
                function_call_details={
                    "tools": None,
                    "example_plans_provided": False,
                    "end_user_provided": False,
                    "plan_inputs_provided": True,
                },
                name="portia_function_call",
            )
        )

        assert plan.plan_context.query == query
        # Should have plan inputs
        assert len(plan.plan_inputs) > 0


@pytest.mark.asyncio
async def test_portia_arun_query(
    portia: Portia, planning_model: MagicMock, telemetry: MagicMock
) -> None:
    """Test async running a query."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )

    plan_run = await portia.arun(query)
    telemetry.capture.assert_called_once_with(
        PortiaFunctionCallTelemetryEvent(
            function_name="portia_arun",
            function_call_details={
                "tools": None,
                "example_plans_provided": False,
                "end_user_provided": False,
                "plan_run_inputs_provided": False,
            },
            name="portia_function_call",
        )
    )

    assert plan_run.state == "COMPLETE"


@pytest.mark.asyncio
async def test_portia_arun_query_tool_list(planning_model: MagicMock, telemetry: MagicMock) -> None:
    """Test async running a query with tool list."""
    query = "example query"
    addition_tool = AdditionTool()
    clarification_tool = ClarificationTool()
    portia = Portia(
        config=get_test_config(
            models=GenerativeModelsConfig(
                planning_model=planning_model,
            ),
        ),
        tools=[addition_tool, clarification_tool],
        telemetry=telemetry,
    )

    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan_run = await portia.arun(query)

    telemetry.capture.assert_called_once_with(
        PortiaFunctionCallTelemetryEvent(
            function_name="portia_arun",
            function_call_details={
                "tools": None,
                "example_plans_provided": False,
                "end_user_provided": False,
                "plan_run_inputs_provided": False,
            },
            name="portia_function_call",
        )
    )
    assert plan_run.state == "COMPLETE"


@pytest.mark.asyncio
async def test_portia_arun_query_disk_storage(planning_model: MagicMock) -> None:
    """Test async running a query with disk storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        query = "example query"
        config = Config.from_default(
            storage_class=StorageClass.DISK,
            openai_api_key=SecretStr("123"),
            storage_dir=tmp_dir,
            models=GenerativeModelsConfig(
                planning_model=planning_model,
            ),
        )
        tool_registry = ToolRegistry([AdditionTool(), ClarificationTool()])
        portia = Portia(config=config, tools=tool_registry)

        planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
        plan_run = await portia.arun(query)

        assert plan_run.state == "COMPLETE"
        # Use Path to check for the files
        plan_files = list(Path(tmp_dir).glob("plan-*.json"))
        run_files = list(Path(tmp_dir).glob("prun-*.json"))

        assert len(plan_files) == 1
        assert len(run_files) == 1


@pytest.mark.asyncio
async def test_portia_arun_with_use_cached_plan_success(portia: Portia) -> None:
    """Test async running with use_cached_plan=True when cached plan exists."""
    query = "example query"

    # Create a cached plan
    cached_plan = Plan(
        plan_context=PlanContext(query=query, tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(cached_plan)

    # Mock the storage.get_plan_by_query to return the cached plan
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", return_value=cached_plan
    ) as mock_get_cached:
        plan_run = await portia.arun(query, use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify the plan run was created from the cached plan
        assert plan_run.plan_id == cached_plan.id
        assert plan_run.state == "COMPLETE"


@pytest.mark.asyncio
async def test_portia_arun_with_use_cached_plan_not_found(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async running with use_cached_plan=True when no cached plan exists."""
    query = "example query"

    # Mock the storage.get_plan_by_query to raise StorageError
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", side_effect=StorageError("No plan found for query")
    ) as mock_get_cached:
        # Mock the planning model to return a successful plan
        planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

        plan_run = await portia.arun(query, use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify a new plan was generated and run
        assert plan_run.state == "COMPLETE"
        assert plan_run.plan_id != "plan-00000000-0000-0000-0000-000000000000"  # Not a default UUID


@pytest.mark.asyncio
async def test_portia_arun_with_use_cached_plan_false(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async running with use_cached_plan=False (default behavior)."""
    query = "example query"

    # Create a cached plan
    cached_plan = Plan(
        plan_context=PlanContext(query=query, tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(cached_plan)

    # Mock the planning model to return a successful plan
    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

    # Mock the storage.get_plan_by_query to ensure it's not called
    with mock.patch.object(portia.storage, "aget_plan_by_query") as mock_get_cached:
        plan_run = await portia.arun(query, use_cached_plan=False)

        # Verify get_plan_by_query was NOT called
        mock_get_cached.assert_not_called()

        # Verify a new plan was generated and run
        assert plan_run.state == "COMPLETE"
        assert plan_run.plan_id != cached_plan.id  # Should be a different plan


@pytest.mark.asyncio
async def test_portia_arun_with_use_cached_plan_and_plan_run_inputs(portia: Portia) -> None:
    """Test async running with use_cached_plan=True and plan run inputs."""
    query = "example query"
    plan_run_inputs = [PlanInput(name="$num_a", value=5)]

    # Create a cached plan
    cached_plan = Plan(
        plan_context=PlanContext(query=query, tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(cached_plan)

    # Mock the storage.get_plan_by_query to return the cached plan
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", return_value=cached_plan
    ) as mock_get_cached:
        plan_run = await portia.arun(query, plan_run_inputs=plan_run_inputs, use_cached_plan=True)

        # Verify get_plan_by_query was called
        mock_get_cached.assert_called_once_with(query)

        # Verify the plan run was created from the cached plan
        assert plan_run.plan_id == cached_plan.id
        assert plan_run.state == "COMPLETE"


@pytest.mark.asyncio
async def test_portia_arun_with_use_cached_plan_storage_error_logging(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test that storage errors are logged when use_cached_plan=True in arun method."""
    query = "example query"

    # Mock the storage.get_plan_by_query to raise StorageError
    with mock.patch.object(
        portia.storage, "aget_plan_by_query", side_effect=StorageError("Test storage error")
    ) as mock_get_cached:
        # Mock the planning model to return a successful plan
        planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

        # Mock the logger to capture warning messages
        with mock.patch("portia.portia.logger") as mock_logger:
            plan_run = await portia.arun(query, use_cached_plan=True)

            # Verify get_plan_by_query was called
            mock_get_cached.assert_called_once_with(query)

            # Verify warning was logged
            mock_logger().warning.assert_called_once_with(
                "Error getting cached plan. Using new plan instead: Test storage error"
            )

            # Verify a new plan was generated and run
            assert plan_run.state == "COMPLETE"


@pytest.mark.asyncio
async def test_portia_arun_error_handling(portia: Portia, planning_model: MagicMock) -> None:
    """Test async running with error handling."""
    query = "example query"

    # Mock the planning model to return an error
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error="could not plan",
    )

    with pytest.raises(PlanError):
        await portia.arun(query)


@pytest.mark.asyncio
async def test_portia_arun_plan(
    portia: Portia, planning_model: MagicMock, telemetry: MagicMock
) -> None:
    """Test that arun_plan calls _acreate_plan_run and _aresume."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan = await portia.aplan(query)

    # Mock the _create_plan_run and _aresume methods
    with (
        mock.patch.object(portia, "_acreate_plan_run") as mock_create_plan_run,
        mock.patch.object(portia, "_aresume") as mock_aresume,
    ):
        mock_plan_run = MagicMock()
        mock_resumed_plan_run = MagicMock()
        mock_create_plan_run.return_value = mock_plan_run
        mock_aresume.return_value = mock_resumed_plan_run

        result = await portia.arun_plan(plan)

        telemetry.capture.assert_has_calls(
            [
                mock.call(
                    PortiaFunctionCallTelemetryEvent(
                        function_name="portia_aplan",
                        function_call_details={
                            "tools": None,
                            "example_plans_provided": False,
                            "end_user_provided": False,
                            "plan_inputs_provided": False,
                        },
                        name="portia_function_call",
                    )
                ),
                mock.call(
                    PortiaFunctionCallTelemetryEvent(
                        function_name="portia_arun_plan",
                        function_call_details={
                            "plan_type": "Plan",
                            "end_user_provided": False,
                            "plan_run_inputs_provided": False,
                        },
                        name="portia_function_call",
                    )
                ),
            ]
        )

        mock_create_plan_run.assert_called_once_with(plan, portia.initialize_end_user(), None)
        mock_aresume.assert_called_once_with(mock_plan_run)
        assert result == mock_resumed_plan_run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan_run_inputs",
    [
        [
            PlanInput(name="$num_a", description="First number to add", value=1),
            PlanInput(name="$num_b", description="Second number to add", value=2),
        ],
        [
            {"name": "$num_a", "description": "First number to add", "value": 1},
            {"name": "$num_b", "description": "Second number to add", "value": 2},
        ],
        {
            "$num_a": 1,
            "$num_b": 2,
        },
        [
            {"incorrect_key": "$num_a", "error": "Error"},
        ],
        "error",
    ],
)
async def test_portia_arun_plan_with_plan_run_inputs(
    portia: Portia,
    plan_run_inputs: list[PlanInput] | list[dict[str, object]] | dict[str, object],
) -> None:
    """Test that arun_plan correctly handles plan inputs in different formats."""
    plan = Plan(
        plan_context=PlanContext(query="Add two numbers", tool_ids=["add_tool"]),
        steps=[
            Step(
                task="Add numbers",
                tool_id="add_tool",
                inputs=[
                    Variable(name="$num_a", description="First number"),
                    Variable(name="$num_b", description="Second number"),
                ],
                output="$result",
            ),
        ],
        plan_inputs=[
            PlanInput(name="$num_a", description="First number to add"),
            PlanInput(name="$num_b", description="Second number to add"),
        ],
    )

    mock_agent = MagicMock()
    # Make execute_async return an awaitable
    mock_agent.execute_async = mock.AsyncMock(return_value=LocalDataValue(value=3))

    if plan_run_inputs == "error" or (
        isinstance(plan_run_inputs, list)
        and isinstance(plan_run_inputs[0], dict)
        and "error" in plan_run_inputs[0]
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            await portia.arun_plan(plan, plan_run_inputs=plan_run_inputs)
        return

    # Mock the get_agent_for_step method to return our mock agent
    with mock.patch.object(portia, "_get_agent_for_step", return_value=mock_agent):
        plan_run = await portia.arun_plan(plan, plan_run_inputs=plan_run_inputs)

    assert plan_run.plan_id == plan.id
    assert len(plan_run.plan_run_inputs) == 2
    assert plan_run.plan_run_inputs["$num_a"].get_value() == 1
    assert plan_run.plan_run_inputs["$num_b"].get_value() == 2
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == 3


@pytest.mark.asyncio
async def test_portia_arun_plan_with_plan_uuid(portia: Portia, telemetry: MagicMock) -> None:
    """Test that arun_plan can retrieve a plan from storage using PlanUUID."""
    # Create a plan and save it to storage
    plan = Plan(
        plan_context=PlanContext(query="Test query", tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(plan)

    # Mock the _create_plan_run and _aresume methods
    with (
        mock.patch.object(portia, "_acreate_plan_run") as mock_create_plan_run,
        mock.patch.object(portia, "_aresume") as mock_aresume,
    ):
        mock_plan_run = MagicMock()
        mock_resumed_plan_run = MagicMock()
        mock_create_plan_run.return_value = mock_plan_run
        mock_aresume.return_value = mock_resumed_plan_run

        # Run the plan using its PlanUUID
        result = await portia.arun_plan(plan.id)

        telemetry.capture.assert_called_with(
            PortiaFunctionCallTelemetryEvent(
                function_name="portia_arun_plan",
                function_call_details={
                    "plan_type": "PlanUUID",
                    "end_user_provided": False,
                    "plan_run_inputs_provided": False,
                },
                name="portia_function_call",
            )
        )

        # Verify the plan was retrieved from storage and used
        mock_create_plan_run.assert_called_once_with(plan, portia.initialize_end_user(), None)
        mock_aresume.assert_called_once_with(mock_plan_run)
        assert result == mock_resumed_plan_run


@pytest.mark.asyncio
async def test_portia_arun_plan_with_new_plan(portia: Portia, planning_model: MagicMock) -> None:
    """Test that arun_plan calls _create_plan_run and _aresume with a new plan."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan = await portia.aplan(query)

    # update the id to functionally make this a new plan
    plan.id = PlanUUID(uuid=uuid4())

    # Mock the _create_plan_run and _aresume methods
    with (
        mock.patch.object(portia, "_acreate_plan_run") as mock_create_plan_run,
        mock.patch.object(portia, "_aresume") as mock_aresume,
    ):
        mock_plan_run = MagicMock()
        mock_resumed_plan_run = MagicMock()
        mock_create_plan_run.return_value = mock_plan_run
        mock_aresume.return_value = mock_resumed_plan_run

        result = await portia.arun_plan(plan)

        mock_create_plan_run.assert_called_once_with(
            plan, EndUser(external_id="portia:default_user"), None
        )

        mock_aresume.assert_called_once_with(mock_plan_run)

        assert result == mock_resumed_plan_run


@pytest.mark.asyncio
async def test_portia_arun_plan_with_uuid(portia: Portia) -> None:
    """Test that arun_plan can retrieve a plan from storage using UUID."""
    # Create a plan and save it to storage
    plan = Plan(
        plan_context=PlanContext(query="Test query", tool_ids=["add_tool"]),
        steps=[],
    )
    await portia.storage.asave_plan(plan)

    # Mock the _create_plan_run and _aresume methods
    with (
        mock.patch.object(portia, "_acreate_plan_run") as mock_create_plan_run,
        mock.patch.object(portia, "_aresume") as mock_aresume,
    ):
        mock_plan_run = MagicMock()
        mock_resumed_plan_run = MagicMock()
        mock_create_plan_run.return_value = mock_plan_run
        mock_aresume.return_value = mock_resumed_plan_run

        # Run the plan using its UUID
        result = await portia.arun_plan(plan.id.uuid)

        # Verify the plan was retrieved from storage and used
        mock_create_plan_run.assert_called_once_with(plan, portia.initialize_end_user(), None)
        mock_aresume.assert_called_once_with(mock_plan_run)
        assert result == mock_resumed_plan_run


@pytest.mark.asyncio
async def test_portia_arun_plan_with_missing_inputs(portia: Portia) -> None:
    """Test that arun_plan raises error when required inputs are missing."""
    required_input1 = PlanInput(name="$required1", description="Required input 1", value="value 1")
    required_input2 = PlanInput(name="$required2", description="Required input 2", value="value 2")

    plan = Plan(
        plan_context=PlanContext(query="Plan requiring inputs", tool_ids=["add_tool"]),
        steps=[
            Step(
                task="Use the required input",
                tool_id="add_tool",
                inputs=[
                    Variable(name="$required1", description="Required value"),
                    Variable(name="$required2", description="Required value"),
                ],
                output="$result",
            ),
        ],
        plan_inputs=[required_input1, required_input2],
    )

    # Try to run the plan without providing required inputs
    with pytest.raises(ValueError):  # noqa: PT011
        await portia.arun_plan(plan, plan_run_inputs=[])

    # Should fail with just one of the two required
    with pytest.raises(ValueError):  # noqa: PT011
        await portia.arun_plan(plan, plan_run_inputs=[required_input1])

    # Should work if we provide both required inputs
    with mock.patch.object(portia, "_aresume") as mock_aresume:
        await portia.arun_plan(
            plan,
            plan_run_inputs=[required_input1, required_input2],
        )
        mock_aresume.assert_called_once()


@pytest.mark.asyncio
async def test_portia_arun_plan_with_extra_input_when_expecting_none(portia: Portia) -> None:
    """Test that arun_plan logs warning when extra inputs are provided."""
    # Create a plan with no inputs
    plan = Plan(
        plan_context=PlanContext(query="Plan with no inputs", tool_ids=["add_tool"]),
        steps=[],
        plan_inputs=[],  # No inputs required
    )

    # Run with input that isn't in the plan's inputs
    extra_input = PlanInput(name="$extra", description="Extra unused input", value="value")
    plan_run = await portia.arun_plan(plan, plan_run_inputs=[extra_input])
    assert plan_run.plan_run_inputs == {}


@pytest.mark.asyncio
async def test_portia_aresume(
    portia: Portia, planning_model: MagicMock, telemetry: MagicMock
) -> None:
    """Test async resuming a plan."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan = await portia.aplan(query)
    plan_run = portia.create_plan_run(plan)
    plan_run = await portia.aresume(plan_run)

    assert telemetry.capture.call_count == 3
    telemetry.capture.assert_called_with(
        PortiaFunctionCallTelemetryEvent(
            function_name="portia_aresume",
            function_call_details={"plan_run_provided": True, "plan_run_id_provided": False},
            name="portia_function_call",
        )
    )

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id


@pytest.mark.asyncio
async def test_portia_aresume_after_interruption(portia: Portia, planning_model: MagicMock) -> None:
    """Test async resuming PlanRun after interruption."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan_run = await portia.arun(query)

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1
    plan_run = await portia.aresume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1


@pytest.mark.asyncio
async def test_portia_set_run_state_to_fail_if_keyboard_interrupt_when_aresume(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test that the run state set to FAILED if a KeyboardInterrupt is raised."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan_run = await portia.arun(query)

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1

    with mock.patch.object(portia, "_aexecute_plan_run", side_effect=KeyboardInterrupt):
        await portia.aresume(plan_run)

    assert plan_run.state == PlanRunState.FAILED


@pytest.mark.asyncio
async def test_portia_aresume_edge_cases(portia: Portia, planning_model: MagicMock) -> None:
    """Test edge cases for async execute."""
    with pytest.raises(ValueError):  # noqa: PT011
        await portia.aresume()

    query = "example query"
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan = await portia.aplan(query)
    plan_run = portia.create_plan_run(plan)

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1
    plan_run = await portia.aresume(plan_run_id=plan_run.id)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1

    with pytest.raises(PlanRunNotFoundError):
        await portia.aresume(plan_run_id=PlanRunUUID())


@pytest.mark.asyncio
async def test_portia_arun_invalid_state(portia: Portia, planning_model: MagicMock) -> None:
    """Test async resuming PlanRun with an invalid state."""
    query = "example query"

    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)
    plan_run = await portia.arun(query)

    # Set invalid state
    plan_run.state = PlanRunState.COMPLETE

    with pytest.raises(InvalidPlanRunStateError):
        await portia.aresume(plan_run)


@pytest.mark.asyncio
async def test_portia_ahandle_clarification(planning_model: MagicMock) -> None:
    """Test that portia can handle a clarification asynchronously."""
    clarification_handler = TestClarificationHandler()
    portia = Portia(
        config=get_test_config(models=GenerativeModelsConfig(planning_model=planning_model)),
        tools=[ClarificationTool()],
        execution_hooks=ExecutionHooks(
            clarification_handler=clarification_handler,
            after_step_execution=MagicMock(),
            after_plan_run=MagicMock(),
            before_step_execution=MagicMock(),
        ),
    )
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[
            Step(
                task="Raise a clarification",
                tool_id="clarification_tool",
                output="$output",
            ),
        ],
        error=None,
    )

    mock_step_agent = mock.MagicMock()
    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.side_effect = "I caught the clarification"

    with (
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan = await portia.aplan("Raise a clarification")
        plan_run = portia.create_plan_run(plan)

        # Create the clarification values with the correct plan_run_id
        clarification_value = LocalDataValue(
            value=InputClarification(
                plan_run_id=plan_run.id,
                user_guidance="Handle this clarification",
                argument_name="raise_clarification",
                source="Test portia handle clarification",
            ),
        )
        final_value = LocalDataValue(value="I caught the clarification")

        mock_step_agent.execute_async = mock.AsyncMock(
            side_effect=[clarification_value, final_value]
        )

        await portia.aresume(plan_run)
        assert plan_run.state == PlanRunState.COMPLETE

        # Check that the clarifications were handled correctly
        assert clarification_handler.received_clarification is not None
        assert (
            clarification_handler.received_clarification.user_guidance
            == "Handle this clarification"
        )
        assert portia.execution_hooks.before_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        assert portia.execution_hooks.after_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        assert portia.execution_hooks.after_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_portia_aerror_clarification(portia: Portia, planning_model: MagicMock) -> None:
    """Test that portia can handle an error clarification asynchronously."""
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan_run = await portia.arun("test query")

    portia.error_clarification(
        ValueConfirmationClarification(
            plan_run_id=plan_run.id,
            user_guidance="Handle this clarification",
            argument_name="raise_clarification",
            source="Test portia error clarification",
        ),
        error=ValueError("test error"),
    )
    assert plan_run.state == PlanRunState.FAILED


@pytest.mark.asyncio
async def test_portia_aerror_clarification_with_plan_run(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test that portia can handle an error clarification with plan run asynchronously."""
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[],
        error=None,
    )
    plan_run = await portia.arun("test query")

    portia.error_clarification(
        ValueConfirmationClarification(
            plan_run_id=plan_run.id,
            user_guidance="Handle this clarification",
            argument_name="raise_clarification",
            source="Test portia error clarification with plan run",
        ),
        error=ValueError("test error"),
        plan_run=plan_run,
    )
    assert plan_run.state == PlanRunState.FAILED


@pytest.mark.asyncio
async def test_portia_aexecute_plan_run_with_introspection_skip(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async execute plan run with introspection agent returning SKIP outcome."""
    # Setup mock plan and response
    step1 = Step(task="Step 1", inputs=[], output="$step1_result", condition="some_condition")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result")
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[step1, step2],
        error=None,
    )

    # Mock introspection agent to return SKIP for first step
    mock_introspection = MagicMock()
    mock_introspection.apre_step_introspection = mock.AsyncMock(
        return_value=PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.SKIP,
            reason="Condition not met",
        )
    )

    # Mock step agent to return output for second step
    mock_step_agent = MagicMock()
    mock_step_agent.execute_async = mock.AsyncMock(
        return_value=LocalDataValue(value="Step 2 result")
    )

    with (
        mock.patch.object(portia, "_get_introspection_agent", return_value=mock_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan_run = await portia.arun("Test query with skipped step")

        # Verify result
        assert plan_run.state == PlanRunState.COMPLETE
        assert "$step1_result" in plan_run.outputs.step_outputs
        assert plan_run.outputs.step_outputs["$step1_result"].get_value() == SKIPPED_OUTPUT
        assert "$step2_result" in plan_run.outputs.step_outputs
        assert plan_run.outputs.step_outputs["$step2_result"].get_value() == "Step 2 result"
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_value() == "Step 2 result"


@pytest.mark.asyncio
async def test_portia_aexecute_plan_run_with_introspection_complete(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test async execute plan run with introspection agent returning COMPLETE outcome."""
    portia.execution_hooks = ExecutionHooks(
        after_plan_run=MagicMock(),
    )

    # Setup mock plan and response
    step1 = Step(task="Step 1", inputs=[], output="$step1_result")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result", condition="some_condition")
    step3 = Step(task="Step 3", inputs=[], output="$step3_result")
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[step1, step2, step3],
        error=None,
    )

    # Mock step agent for first step
    mock_step_agent = MagicMock()
    mock_step_agent.execute_async = mock.AsyncMock(
        return_value=LocalDataValue(value="Step 1 result")
    )

    # Configure the COMPLETE outcome for the introspection agent
    mock_introspection_complete = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.COMPLETE,
        reason="Remaining steps cannot be executed",
    )

    final_output = LocalDataValue(
        value="Step 1 result",
        summary="Execution completed early",
    )

    async def custom_handle_introspection(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        plan_run = kwargs.get("plan_run")  # type: ignore  # noqa: PGH003

        if plan_run is not None and plan_run.current_step_index == 1:
            plan_run.outputs.step_outputs["$step2_result"] = LocalDataValue(
                value=COMPLETED_OUTPUT,
                summary="Remaining steps cannot be executed",
            )
            plan_run.outputs.final_output = final_output
            plan_run.state = PlanRunState.COMPLETE

            return (plan_run, mock_introspection_complete)

        # Otherwise continue normally
        return (
            plan_run,
            PreStepIntrospection(
                outcome=PreStepIntrospectionOutcome.CONTINUE,
                reason="Condition met",
            ),
        )

    with (
        mock.patch.object(portia, "_agenerate_introspection_outcome", custom_handle_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        # Run the test
        plan_run = await portia.arun("Test query with early completed execution")

        # Verify result based on our simulated outcomes
        assert plan_run.state == PlanRunState.COMPLETE
        assert "$step2_result" in plan_run.outputs.step_outputs
        assert plan_run.outputs.step_outputs["$step2_result"].get_value() == COMPLETED_OUTPUT
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.get_summary() == "Execution completed early"
        assert portia.execution_hooks.after_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        portia.execution_hooks.after_plan_run.assert_called_once_with(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
            mock.ANY, mock.ANY, final_output
        )


@pytest.mark.asyncio
async def test_portia_agenerate_introspection_outcome_complete(portia: Portia) -> None:
    """Test the actual implementation of _agenerate_introspection_outcome for COMPLETE outcome."""
    # Create a plan with conditions
    step = Step(task="Test step", inputs=[], output="$test_output", condition="some_condition")
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        end_user_id="test123",
        current_step_index=0,
        state=PlanRunState.IN_PROGRESS,
    )

    mock_introspection = MagicMock()
    mock_introspection.apre_step_introspection = mock.AsyncMock(
        return_value=PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.COMPLETE,
            reason="Stopping execution",
        )
    )

    # Mock the _get_final_output method to return a predefined output
    mock_final_output = LocalDataValue(value="Final result", summary="Final summary")
    with mock.patch.object(portia, "_get_final_output", return_value=mock_final_output):
        # Call the actual method (not mocked)
        previous_output = LocalDataValue(value="Previous step result")
        updated_plan_run, outcome = await portia._agenerate_introspection_outcome(
            introspection_agent=mock_introspection,
            plan=plan,
            plan_run=plan_run,
            last_executed_step_output=previous_output,
        )

        # Verify the outcome
        assert outcome.outcome == PreStepIntrospectionOutcome.COMPLETE
        assert outcome.reason == "Stopping execution"

        # Verify plan_run was updated correctly
        assert updated_plan_run.outputs.step_outputs["$test_output"].get_value() == COMPLETED_OUTPUT
        assert (
            updated_plan_run.outputs.step_outputs["$test_output"].get_summary()
            == "Stopping execution"
        )
        assert updated_plan_run.outputs.final_output == mock_final_output
        assert updated_plan_run.state == PlanRunState.COMPLETE


@pytest.mark.asyncio
async def test_portia_agenerate_introspection_outcome_skip(portia: Portia) -> None:
    """Test the actual implementation of _agenerate_introspection_outcome for SKIP outcome."""
    # Create a plan with conditions
    step = Step(task="Test step", inputs=[], output="$test_output", condition="some_condition")
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=0,
        end_user_id="test123",
        state=PlanRunState.IN_PROGRESS,
    )

    mock_introspection = MagicMock()
    mock_introspection.apre_step_introspection = mock.AsyncMock(
        return_value=PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.SKIP,
            reason="Skipping step",
        )
    )

    previous_output = LocalDataValue(value="Previous step result")
    updated_plan_run, outcome = await portia._agenerate_introspection_outcome(
        introspection_agent=mock_introspection,
        plan=plan,
        plan_run=plan_run,
        last_executed_step_output=previous_output,
    )

    assert outcome.outcome == PreStepIntrospectionOutcome.SKIP
    assert outcome.reason == "Skipping step"

    assert updated_plan_run.outputs.step_outputs["$test_output"].get_value() == SKIPPED_OUTPUT
    assert updated_plan_run.outputs.step_outputs["$test_output"].get_summary() == "Skipping step"
    assert updated_plan_run.state == PlanRunState.IN_PROGRESS  # State should remain IN_PROGRESS


@pytest.mark.asyncio
async def test_portia_agenerate_introspection_outcome_no_condition(portia: Portia) -> None:
    """Test _agenerate_introspection_outcome when step has no condition."""
    # Create a plan with a step that has no condition
    step = Step(task="Test step", inputs=[], output="$test_output")  # No condition
    plan = Plan(
        plan_context=PlanContext(query="test query", tool_ids=[]),
        steps=[step],
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=0,
        end_user_id="test123",
        state=PlanRunState.IN_PROGRESS,
    )

    # Mock the introspection agent (should not be called)
    mock_introspection = MagicMock()

    # Call the actual method
    previous_output = LocalDataValue(value="Previous step result")
    updated_plan_run, outcome = await portia._agenerate_introspection_outcome(
        introspection_agent=mock_introspection,
        plan=plan,
        plan_run=plan_run,
        last_executed_step_output=previous_output,
    )

    # Verify default outcome is CONTINUE
    assert outcome.outcome == PreStepIntrospectionOutcome.CONTINUE
    assert outcome.reason == "No condition to evaluate."

    # The introspection agent should not be called
    mock_introspection.apre_step_introspection.assert_not_called()

    # Plan run should be unchanged (no step outputs added)
    assert "$test_output" not in updated_plan_run.outputs.step_outputs
    assert updated_plan_run.state == PlanRunState.IN_PROGRESS


@pytest.mark.asyncio
async def test_portia_aexecute_step_hooks(portia: Portia, planning_model: MagicMock) -> None:
    """Test that all execution step hooks are called in correct order with right arguments."""
    execution_hooks = ExecutionHooks(
        before_plan_run=MagicMock(),
        before_step_execution=MagicMock(),
        after_step_execution=MagicMock(),
        after_plan_run=MagicMock(),
    )
    portia.execution_hooks = execution_hooks

    # Create a plan with two steps
    step1 = Step(task="Step 1", tool_id="add_tool", output="$step1_result")
    step2 = Step(task="Step 2", tool_id="add_tool", output="$step2_result")
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[step1, step2], error=None
    )

    mock_agent = MagicMock()
    step_1_result = LocalDataValue(value="Step 1 result")
    step_2_result = LocalDataValue(value="Step 2 result")
    mock_agent.execute_async = mock.AsyncMock(side_effect=[step_1_result, step_2_result])
    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.return_value = None

    with (
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_agent),
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
    ):
        plan_run = await portia.arun("Test execution hooks")

    assert plan_run.state == PlanRunState.COMPLETE

    assert execution_hooks.before_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.before_step_execution.call_count == 2  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_step_execution.call_count == 2  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]

    plan = await portia.storage.aget_plan(plan_run.plan_id)
    execution_hooks.before_plan_run.assert_called_once_with(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        ReadOnlyPlan.from_plan(plan), mock.ANY
    )

    execution_hooks.before_step_execution.assert_any_call(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        ReadOnlyPlan.from_plan(plan), mock.ANY, ReadOnlyStep.from_step(step1)
    )
    execution_hooks.before_step_execution.assert_any_call(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        ReadOnlyPlan.from_plan(plan), mock.ANY, ReadOnlyStep.from_step(step2)
    )

    execution_hooks.after_step_execution.assert_any_call(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        ReadOnlyPlan.from_plan(plan), mock.ANY, ReadOnlyStep.from_step(step1), step_1_result
    )
    execution_hooks.after_step_execution.assert_any_call(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        ReadOnlyPlan.from_plan(plan), mock.ANY, ReadOnlyStep.from_step(step2), step_2_result
    )

    execution_hooks.after_plan_run.assert_called_once_with(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        ReadOnlyPlan.from_plan(plan), mock.ANY, step_2_result
    )


@pytest.mark.asyncio
async def test_portia_aresume_with_skipped_steps(portia: Portia) -> None:
    """Test async resuming a plan run with skipped steps and verifying final output.

    This test verifies:
    1. Resuming from a middle index works correctly
    2. Steps marked as SKIPPED are properly skipped during execution
    3. The final output is correctly computed from the last non-SKIPPED step
    """
    # Create a plan with multiple steps
    step1 = Step(task="Step 1", inputs=[], output="$step1_result")
    step2 = Step(task="Step 2", inputs=[], output="$step2_result", condition="true")
    step3 = Step(task="Step 3", inputs=[], output="$step3_result", condition="false")
    step4 = Step(task="Step 4", inputs=[], output="$step4_result", condition="false")
    plan = Plan(
        plan_context=PlanContext(query="Test query with skips", tool_ids=[]),
        steps=[step1, step2, step3, step4],
    )

    # Create a plan run that's partially completed (step1 is done)
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=1,  # Resume from step 2
        state=PlanRunState.IN_PROGRESS,
        end_user_id="test123",
        outputs=PlanRunOutputs(
            step_outputs={
                "$step1_result": LocalDataValue(value="Step 1 result", summary="Summary of step 1"),
            },
        ),
    )

    # Mock the storage to return our plan
    await portia.storage.asave_plan(plan)
    await portia.storage.asave_plan_run(plan_run)

    # Mock introspection agent to SKIP steps 3 and 4
    mock_introspection = MagicMock()

    async def mock_introspection_outcome(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        plan_run = kwargs.get("plan_run")
        if plan_run.current_step_index in (2, 3):  # pyright: ignore[reportOptionalMemberAccess] # Skip both step3 and step4
            return PreStepIntrospection(
                outcome=PreStepIntrospectionOutcome.SKIP,
                reason="Condition is false",
            )
        return PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.CONTINUE,
            reason="Continue execution",
        )

    mock_introspection.apre_step_introspection = mock.AsyncMock(
        side_effect=mock_introspection_outcome
    )

    # Mock step agent to return expected output for step 2 only (steps 3 and 4 will be skipped)
    mock_step_agent = MagicMock()
    mock_step_agent.execute_async = mock.AsyncMock(
        return_value=LocalDataValue(
            value="Step 2 result",
            summary="Summary of step 2",
        )
    )

    # Mock the final output summarizer
    expected_summary = "Combined summary of steps 1 and 2"
    mock_summarizer = MagicMock()
    mock_summarizer.create_summary.return_value = expected_summary

    with (
        mock.patch.object(portia, "_get_introspection_agent", return_value=mock_introspection),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
        mock.patch("portia.portia.FinalOutputSummarizer", return_value=mock_summarizer),
    ):
        result_plan_run = await portia.aresume(plan_run)

        assert result_plan_run.state == PlanRunState.COMPLETE

        assert result_plan_run.outputs.step_outputs["$step1_result"].get_value() == "Step 1 result"
        assert result_plan_run.outputs.step_outputs["$step2_result"].get_value() == "Step 2 result"
        assert result_plan_run.outputs.step_outputs["$step3_result"].get_value() == SKIPPED_OUTPUT
        assert result_plan_run.outputs.step_outputs["$step4_result"].get_value() == SKIPPED_OUTPUT
        assert result_plan_run.outputs.final_output is not None
        assert result_plan_run.outputs.final_output.get_value() == "Step 2 result"
        assert result_plan_run.outputs.final_output.get_summary() == expected_summary
        assert result_plan_run.current_step_index == 3


@pytest.mark.asyncio
async def test_portia_aexecute_step_hooks_with_error(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test that execution hooks are called correctly when an error occurs."""
    execution_hooks = ExecutionHooks(
        before_plan_run=MagicMock(),
        before_step_execution=MagicMock(),
        after_step_execution=MagicMock(),
        after_plan_run=MagicMock(),
    )
    portia.execution_hooks = execution_hooks

    step1 = Step(task="Step 1", tool_id="add_tool", output="$step1_result")
    planning_model.aget_structured_response.return_value = StepsOrError(steps=[step1], error=None)

    # Mock the first agent to raise an error
    mock_agent = MagicMock()
    mock_agent.execute_async.side_effect = ValueError("Test execution error")

    with mock.patch.object(portia, "_get_agent_for_step", return_value=mock_agent):
        plan_run = await portia.arun("Test execution hooks with error")
    assert plan_run.state == PlanRunState.FAILED

    assert execution_hooks.before_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.before_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert execution_hooks.after_plan_run.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_portia_aexecute_step_hooks_with_skip(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test that execution hooks can skip steps when before_step_execution returns SKIP."""
    step1 = Step(task="Step 1", tool_id="add_tool", output="$step1_result")
    step2 = Step(task="Step 2", tool_id="add_tool", output="$step2_result")
    planning_model.aget_structured_response.return_value = StepsOrError(
        steps=[step1, step2], error=None
    )

    # Create execution hooks that will skip the first step
    execution_hooks = ExecutionHooks(
        before_step_execution=lambda plan, plan_run, step: (  # noqa: ARG005
            BeforeStepExecutionOutcome.SKIP
            if step.task == "Step 1"
            else BeforeStepExecutionOutcome.CONTINUE
        ),
        after_step_execution=MagicMock(),
    )
    portia.execution_hooks = execution_hooks

    mock_agent = MagicMock()
    step_2_result = LocalDataValue(value="Step 2 result")
    mock_agent.execute_async = mock.AsyncMock(return_value=step_2_result)

    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.return_value = None

    with (
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_agent),
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
    ):
        plan_run = await portia.arun("Test execution hooks with skip")

    assert plan_run.state == PlanRunState.COMPLETE

    assert "$step1_result" not in plan_run.outputs.step_outputs
    assert "$step2_result" in plan_run.outputs.step_outputs
    assert plan_run.outputs.step_outputs["$step2_result"].get_value() == "Step 2 result"

    assert execution_hooks.after_step_execution.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    execution_hooks.after_step_execution.assert_called_once_with(  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        mock.ANY, mock.ANY, ReadOnlyStep.from_step(step2), step_2_result
    )


@pytest.mark.asyncio
async def test_portia_aexecute_step_hooks_after_step_exception(
    portia: Portia, planning_model: MagicMock
) -> None:
    """Test after_step_execution hook exception handling asynchronously."""

    def failing_after_step_hook(plan, plan_run, step, output):  # noqa: ANN202, ARG001, ANN001
        raise ValueError("Test after_step_execution hook exception")

    execution_hooks = ExecutionHooks(
        after_step_execution=failing_after_step_hook,
    )
    portia.execution_hooks = execution_hooks

    step1 = Step(task="Step 1", tool_id="add_tool", output="$step1_result")
    planning_model.aget_structured_response.return_value = StepsOrError(steps=[step1], error=None)
    mock_agent = MagicMock()
    step_1_result = LocalDataValue(value="Step 1 result")
    mock_agent.execute_async = mock.AsyncMock(return_value=step_1_result)

    with mock.patch.object(portia, "_get_agent_for_step", return_value=mock_agent):
        plan_run = await portia.arun("Test after_step_execution hook exception")

    assert plan_run.state == PlanRunState.FAILED
    assert plan_run.outputs.final_output.get_value() == "Test after_step_execution hook exception"  # pyright: ignore[reportOptionalMemberAccess]


class ReadyTool(Tool):
    """A dummy tool that can be set to ready or not ready."""

    id: str = "ready_tool"
    name: str = "Ready Tool"
    description: str = "A dummy tool that can be set to ready or not ready."
    args_schema: type[BaseModel] = _ArgsSchemaPlaceholder
    output_schema: tuple[str, str] = ("ReadyResponse", "A response from the tool")

    auth_url: str = "https://fake.portiaai.test/auth"
    is_ready: bool | list[bool] = False

    def _get_clarifications(self, plan_run_id: PlanRunUUID) -> list[Clarification]:
        """Get clarifications for this tool."""
        if not self.is_ready:
            return [
                ActionClarification(
                    user_guidance="Please authenticate",
                    plan_run_id=plan_run_id,
                    action_url=HttpUrl(self.auth_url),
                )
            ]
        return []

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:
        """Check if the tool is ready."""
        clarifications = self._get_clarifications(ctx.plan_run.id)
        return ReadyResponse(
            ready=len(clarifications) == 0,
            clarifications=clarifications,
        )

    def run(self, ctx: ToolRunContext) -> None:
        """Run the tool."""


@pytest.mark.asyncio
async def test_portia_acustom_tool_ready_not_ready() -> None:
    """Test clarification handling for a custom tool that is not ready."""
    ready_tool = ReadyTool()
    portia = Portia(
        config=get_test_config(),
        tools=[ready_tool],
    )
    plan = PlanBuilder().step("", ready_tool.id).build()
    plan_run = await portia.acreate_plan_run(plan, end_user="123")
    await portia.storage.asave_plan(plan)  # Explicitly save plan for test

    output_plan_run = await portia.aresume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 1
    outstanding_clarification = output_plan_run.get_outstanding_clarifications()[0]
    assert isinstance(outstanding_clarification, ActionClarification)
    assert outstanding_clarification.resolved is False
    assert outstanding_clarification.plan_run_id == plan_run.id
    assert str(outstanding_clarification.action_url) == ready_tool.auth_url


@pytest.mark.asyncio
async def test_portia_acustom_tool_ready_resume_multiple_instances_of_same_tool() -> None:
    """Clarification handling for multiple instances of the same tool with custom implementation.

    Only one clarification should be raised for the tool.
    """
    ready_tool = ReadyTool()
    portia = Portia(
        config=get_test_config(),
        tools=[ready_tool, ready_tool],
    )
    plan = PlanBuilder().step("1", ready_tool.id).step("2", ready_tool.id).build()
    plan_run = await portia.acreate_plan_run(plan, end_user="123")
    await portia.storage.asave_plan(plan)  # Explicitly save plan for test

    output_plan_run = await portia.aresume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 1
    outstanding_clarification = output_plan_run.get_outstanding_clarifications()[0]
    assert isinstance(outstanding_clarification, ActionClarification)
    assert outstanding_clarification.resolved is False
    assert outstanding_clarification.plan_run_id == plan_run.id
    assert str(outstanding_clarification.action_url) == ready_tool.auth_url


@pytest.mark.asyncio
async def test_portia_acustom_tool_ready_resume_multiple_custom_tools() -> None:
    """Test clarifications are raised for multiple tools in a plan run if they require it."""
    ready_tool = ReadyTool(id="ready_tool", auth_url="https://fake.portiaai.test/auth")
    ready_tool_2 = ReadyTool(id="ready_tool_2", auth_url="https://fake.portiaai.test/auth2")
    portia = Portia(config=get_test_config(), tools=[ready_tool, ready_tool_2])
    plan = PlanBuilder().step("1", ready_tool.id).step("2", ready_tool_2.id).build()
    plan_run = await portia.acreate_plan_run(plan, end_user="123")
    await portia.storage.asave_plan(plan)  # Explicitly save plan for test

    output_plan_run = await portia.aresume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 2
    outstanding_clarifications = output_plan_run.get_outstanding_clarifications()
    assert isinstance(outstanding_clarifications[0], ActionClarification)
    assert outstanding_clarifications[0].plan_run_id == plan_run.id
    assert str(outstanding_clarifications[0].action_url) == ready_tool.auth_url
    assert outstanding_clarifications[0].resolved is False
    assert isinstance(outstanding_clarifications[1], ActionClarification)
    assert outstanding_clarifications[1].plan_run_id == plan_run.id
    assert str(outstanding_clarifications[1].action_url) == ready_tool_2.auth_url
    assert outstanding_clarifications[1].resolved is False


@pytest.mark.asyncio
async def test_portia_arun_plan_with_all_plan_types_error_handling(portia: Portia) -> None:
    """Test error handling for all plan ID types when plan doesn't exist."""
    # Plan objects are automatically saved to storage, so they never raise PlanNotFoundError
    # Only PlanUUID can raise PlanNotFoundError

    # Test with non-existent PlanUUID
    with pytest.raises(PlanNotFoundError):
        await portia.arun_plan(PlanUUID.from_string("plan-99fc470b-4cbd-489b-b251-7076bf7e8f05"))

    # Test that Plan objects work (they get auto-saved, no error)
    non_existent_plan = Plan(
        plan_context=PlanContext(query="non-existent", tool_ids=["add_tool"]),
        steps=[Step(task="Task", tool_id="add_tool", inputs=[], output="$result")],
    )
    # This should succeed because Plan objects are auto-saved
    with mock.patch.object(portia, "_aresume") as mock_aresume:
        mock_aresume.side_effect = lambda x: x
        plan_run = await portia.arun_plan(non_existent_plan)
        assert plan_run.plan_id == non_existent_plan.id


@pytest.mark.asyncio
async def test_portia_aexecute_plan_run_and_handle_clarifications_keyboard_interrupt(
    portia: Portia,
) -> None:
    """Test that KeyboardInterrupt is handled correctly in async version."""
    plan, plan_run = get_test_plan_run()

    with (
        mock.patch.object(portia, "_aexecute_plan_run", side_effect=KeyboardInterrupt()),
        mock.patch.object(portia.storage, "save_plan_run"),
    ):
        result = await portia.aexecute_plan_run_and_handle_clarifications(plan, plan_run)

        assert result.state == PlanRunState.FAILED
