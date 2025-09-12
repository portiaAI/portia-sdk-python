"""Test the single tool agent step."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.reference import Input, StepOutput
from portia.builder.single_tool_agent_step import SingleToolAgentStep
from portia.config import ExecutionAgentType
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.plan import Step as PlanStep
from portia.run_context import StepOutputValue
from portia.tool import Tool
from portia.tool_registry import ToolRegistry


class MockOutputSchema(BaseModel):
    """Mock output schema for testing."""

    result: str
    count: int


class MockTool(Tool[str]):
    """Mock tool for testing."""

    def __init__(self) -> None:
        """Initialize mock tool."""
        super().__init__(
            id="mock_tool",
            name="Mock Tool",
            description="A mock tool for testing",
            output_schema=("str", "Mock result string"),
        )

    def run(self, ctx: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG002
        """Run the mock tool."""
        return "mock result"

    async def arun(self, ctx: Any, **kwargs: Any) -> str:  # noqa: ANN401, ARG002
        """Run the mock tool."""
        return "mock result"


def test_single_tool_agent_initialization() -> None:
    """Test SingleToolAgent initialization."""
    inputs = [Input("query"), "context"]
    step = SingleToolAgentStep(
        task="Search for information",
        tool="search_tool",
        step_name="search_agent",
        inputs=inputs,
        output_schema=MockOutputSchema,
    )

    assert step.task == "Search for information"
    assert step.tool == "search_tool"
    assert step.inputs == inputs
    assert step.output_schema == MockOutputSchema


def test_single_tool_agent_str() -> None:
    """Test SingleToolAgent str method."""
    step = SingleToolAgentStep(
        task="Search for info",
        tool="search_tool",
        step_name="search",
    )

    assert str(step) == "SingleToolAgentStep(task='Search for info', tool='search_tool')"


def test_single_tool_agent_str_with_output_schema() -> None:
    """Test SingleToolAgent str method with output schema."""
    step = SingleToolAgentStep(
        task="Search for info",
        tool="search_tool",
        step_name="search",
        output_schema=MockOutputSchema,
    )

    expected_str = (
        "SingleToolAgentStep(task='Search for info', tool='search_tool' -> MockOutputSchema)"
    )
    assert str(step) == expected_str


def test_single_tool_agent_with_tool_object() -> None:
    """Test SingleToolAgentStep with a Tool instance."""
    tool_instance = MockTool()
    step = SingleToolAgentStep(task="Use tool", tool=tool_instance, step_name="use")

    assert str(step) == "SingleToolAgentStep(task='Use tool', tool='mock_tool')"

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$use_output"
    legacy_step = step.to_legacy_step(mock_plan)
    assert legacy_step.tool_id == "mock_tool"


def test_single_tool_agent_step_adds_tool_to_registry() -> None:
    """Ensure SingleToolAgentStep registers Tool objects."""
    tool_instance = MockTool()
    step = SingleToolAgentStep(task="Use tool", tool=tool_instance, step_name="use")
    run_data = Mock()
    run_data.config.execution_agent_type = ExecutionAgentType.DEFAULT
    run_data.tool_registry = ToolRegistry()
    run_data.storage = Mock()
    run_data.plan_run = Mock()
    run_data.legacy_plan = Mock()
    run_data.end_user = Mock()
    run_data.execution_hooks = Mock()

    with (
        patch("portia.builder.single_tool_agent_step.ToolCallWrapper") as mock_wrapper,
        patch("portia.builder.single_tool_agent_step.DefaultExecutionAgent") as mock_agent,
    ):
        mock_wrapper.return_value = Mock()
        mock_agent.return_value = Mock()
        step._get_agent_for_step(run_data)

    assert tool_instance.id in run_data.tool_registry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution_agent_type", "expected_one_shot"),
    [
        (ExecutionAgentType.ONE_SHOT, True),
        (ExecutionAgentType.DEFAULT, False),
    ],
)
async def test_single_tool_agent_step_with_execution_agent_types(
    execution_agent_type: ExecutionAgentType, expected_one_shot: bool
) -> None:
    """Test SingleToolAgentStep uses correct execution agent type."""
    step = SingleToolAgentStep(tool="mock_tool", task="test task", step_name="agent_step")

    # Set up mock run_data with config
    mock_run_data = Mock()
    mock_run_data.config.execution_agent_type = execution_agent_type
    mock_run_data.tool_registry = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.legacy_plan = Mock()
    mock_run_data.end_user = Mock()
    mock_run_data.execution_hooks = Mock()

    mock_tool = Mock()
    mock_output = LocalDataValue(value="Agent execution result")

    with (
        patch(
            "portia.builder.single_tool_agent_step.ToolCallWrapper.from_tool_id"
        ) as mock_get_tool,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(
            OneShotAgent, "execute_async", new_callable=AsyncMock, return_value=mock_output
        ) as mock_oneshot_execute,
        patch.object(
            DefaultExecutionAgent,
            "execute_async",
            new_callable=AsyncMock,
            return_value=mock_output,
        ) as mock_default_execute,
    ):
        mock_get_tool.return_value = mock_tool
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, LocalDataValue)
        assert result.value == "Agent execution result"

        if expected_one_shot:
            mock_oneshot_execute.assert_called_once()
            mock_default_execute.assert_not_called()
        else:
            mock_default_execute.assert_called_once()
            mock_oneshot_execute.assert_not_called()


def test_single_tool_agent_to_legacy_step() -> None:
    """Test SingleToolAgent to_legacy_step method."""
    inputs = [Input("query"), StepOutput(0)]
    step = SingleToolAgentStep(
        task="Search for information",
        tool="search_tool",
        step_name="search_agent",
        inputs=inputs,
        output_schema=MockOutputSchema,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$search_agent_output"

    with (
        patch.object(inputs[0], "get_legacy_name") as mock_input_name,
        patch.object(inputs[1], "get_legacy_name") as mock_step_output_name,
    ):
        mock_input_name.return_value = "query"
        mock_step_output_name.return_value = "step_0_output"

        legacy_step = step.to_legacy_step(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Search for information"
        assert legacy_step.tool_id == "search_tool"
        assert legacy_step.output == "$search_agent_output"
        assert legacy_step.structured_output_schema == MockOutputSchema

        assert len(legacy_step.inputs) == 2
        assert legacy_step.inputs[0].name == "query"
        assert legacy_step.inputs[1].name == "step_0_output"


@pytest.mark.asyncio
async def test_single_tool_agent_with_string_template_task_and_inputs() -> None:
    """Test SingleToolAgentStep with string templates in task and inputs."""
    step = SingleToolAgentStep(
        task=f"Search for information about {StepOutput(0)} requested by {Input('username')}",
        tool="search_tool",
        step_name="templated_search",
        inputs=[f"Context: {StepOutput(1)}", "Additional info", Input("category")],
    )
    mock_run_data = Mock()
    mock_run_data.config.execution_agent_type = ExecutionAgentType.ONE_SHOT
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="username"),
        PlanInput(name="category"),
    ]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {
        "username": LocalDataValue(value="Alice"),
        "category": LocalDataValue(value="Technology"),
    }
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0,
            step_name="step0",
            value="machine learning",
            description="step0",
        ),
        StepOutputValue(
            step_num=1,
            step_name="step1",
            value="AI research",
            description="step1",
        ),
    ]

    # Create mock agent and output object
    mock_output_obj = LocalDataValue(value="Search completed successfully")
    mock_tool = Mock()

    # Mock the plan for to_legacy_step conversion
    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$templated_search_output"
    mock_run_data.plan = mock_plan

    with (
        patch(
            "portia.builder.single_tool_agent_step.ToolCallWrapper.from_tool_id"
        ) as mock_get_tool,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(
            OneShotAgent, "execute_async", new_callable=AsyncMock, return_value=mock_output_obj
        ),
    ):
        mock_get_tool.return_value = mock_tool
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, LocalDataValue)
        assert result.value == "Search completed successfully"

        # The task should contain the original template string (not resolved yet)
        # Template resolution happens within the execution agent
        expected_task = (
            f"Search for information about {StepOutput(0)} requested by {Input('username')}"
        )
        assert step.task == expected_task
