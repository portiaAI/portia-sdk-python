"""Async tests for portia classes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

from portia.errors import PlanError
from portia.plan import Plan, PlanContext, PlanInput
from portia.planning_agents.base_planning_agent import StepsOrError
from portia.storage import StorageError
from portia.telemetry.views import PortiaFunctionCallTelemetryEvent

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from portia.portia import Portia


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
    portia.storage.save_plan(cached_plan)

    # Mock the storage.get_plan_by_query to return the cached plan
    with mock.patch.object(
        portia.storage, "get_plan_by_query", return_value=cached_plan
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
        portia.storage, "get_plan_by_query", side_effect=StorageError("No plan found for query")
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
    portia.storage.save_plan(cached_plan)

    # Mock the planning model to return a successful plan
    planning_model.aget_structured_response.return_value = StepsOrError(steps=[], error=None)

    # Mock the storage.get_plan_by_query to ensure it's not called
    with mock.patch.object(portia.storage, "get_plan_by_query") as mock_get_cached:
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
    portia.storage.save_plan(cached_plan)

    # Mock the storage.get_plan_by_query to return the cached plan
    with mock.patch.object(
        portia.storage, "get_plan_by_query", return_value=cached_plan
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
