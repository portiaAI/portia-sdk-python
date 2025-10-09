"""Test the react agent step."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.react_agent_step import ReActAgentStep
from portia.builder.reference import Input, StepOutput
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.plan import Step as PlanStep
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


def test_react_agent_step_initialization() -> None:
    """Test ReActAgentStep initialization."""
    inputs = [Input("query"), "context"]
    tools = ["search_tool", "calculator_tool"]
    step = ReActAgentStep(
        task="Research and calculate",
        tools=tools,
        step_name="react_agent",
        inputs=inputs,
        output_schema=MockOutputSchema,
        tool_call_limit=50,
        allow_agent_clarifications=True,
    )

    assert step.task == "Research and calculate"
    assert step.tools == tools
    assert step.inputs == inputs
    assert step.output_schema == MockOutputSchema
    assert step.step_name == "react_agent"
    assert step.tool_call_limit == 50
    assert step.allow_agent_clarifications is True


def test_react_agent_step_initialization_defaults() -> None:
    """Test ReActAgentStep initialization with default values."""
    step = ReActAgentStep(
        task="Simple task",
        tools=["tool1"],
        step_name="simple_react",
    )

    assert step.task == "Simple task"
    assert step.tools == ["tool1"]
    assert step.inputs == []
    assert step.output_schema is None
    assert step.tool_call_limit == 25  # default value
    assert step.allow_agent_clarifications is False  # default value


def test_react_agent_step_str() -> None:
    """Test ReActAgentStep str method."""
    tools = ["tool1", "tool2"]
    step = ReActAgentStep(
        task="Multi-tool task",
        tools=tools,
        step_name="react",
    )

    assert str(step) == f"ReActAgentStep(task='Multi-tool task', tools='{tools}', )"


def test_react_agent_step_str_with_output_schema() -> None:
    """Test ReActAgentStep str method with output schema."""
    tools = ["tool1"]
    step = ReActAgentStep(
        task="Task with schema",
        tools=tools,
        step_name="react",
        output_schema=MockOutputSchema,
    )

    expected_str = f"ReActAgentStep(task='Task with schema', tools='{tools}',  -> MockOutputSchema)"
    assert str(step) == expected_str


def test_react_agent_step_with_tool_objects() -> None:
    """Test ReActAgentStep accepts Tool instances."""
    tools = [MockTool(), "calculator_tool"]
    step = ReActAgentStep(task="Multi-tool task", tools=tools, step_name="react")

    tools_str = "['mock_tool', 'calculator_tool']"
    assert str(step) == f"ReActAgentStep(task='Multi-tool task', tools='{tools_str}', )"

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$react_output"
    legacy_step = step.to_step_data(mock_plan)
    assert legacy_step.tool_id == "mock_tool,calculator_tool"


def test_react_agent_step_adds_tool_to_registry() -> None:
    """Ensure ReActAgentStep registers Tool objects."""
    tool_instance = MockTool()
    step = ReActAgentStep(task="Multi-tool", tools=[tool_instance], step_name="test_step")
    run_data = Mock()
    run_data.tool_registry = ToolRegistry()
    run_data.storage = Mock()
    run_data.plan_run = Mock()

    with (
        patch("portia.builder.react_agent_step.ToolCallWrapper.from_tool_id") as mock_wrapper,
        patch("portia.builder.react_agent_step.ReActAgent") as mock_agent,
    ):
        mock_wrapper.return_value = Mock()
        mock_agent.return_value = Mock()
        step._get_agent_for_step(run_data)

    assert tool_instance.id in run_data.tool_registry


def test_react_agent_step_to_step_data() -> None:
    """Test ReActAgentStep to_step_data method."""
    inputs = [Input("query"), StepOutput(0)]
    tools = ["search_tool", "calculator_tool"]
    step = ReActAgentStep(
        task="Research and calculate",
        tools=tools,
        step_name="react_agent",
        inputs=inputs,
        output_schema=MockOutputSchema,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$react_agent_output"

    with (
        patch.object(inputs[0], "get_legacy_name") as mock_input_name,
        patch.object(inputs[1], "get_legacy_name") as mock_step_output_name,
    ):
        mock_input_name.return_value = "query"
        mock_step_output_name.return_value = "step_0_output"

        legacy_step = step.to_step_data(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Research and calculate"
        assert legacy_step.tool_id == "search_tool,calculator_tool"
        assert legacy_step.output == "$react_agent_output"
        assert legacy_step.structured_output_schema == MockOutputSchema

        assert len(legacy_step.inputs) == 2
        assert legacy_step.inputs[0].name == "query"
        assert legacy_step.inputs[1].name == "step_0_output"


@pytest.mark.asyncio
async def test_react_agent_step_run() -> None:
    """Test ReActAgentStep run method."""
    tools = ["search_tool", "calculator_tool"]
    mock_model = Mock()
    step = ReActAgentStep.model_construct(
        task="Research and calculate",
        tools=tools,
        step_name="react_agent",
        inputs=["context"],
        tool_call_limit=30,
        allow_agent_clarifications=True,
        model=mock_model,
    )

    mock_run_data = Mock()

    mock_tool1 = Mock()
    mock_tool2 = Mock()
    mock_output = LocalDataValue(value="ReAct agent execution result")

    with (
        patch("portia.builder.react_agent_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch("portia.builder.react_agent_step.ReActAgent") as mock_react_agent_class,
    ):
        mock_get_tool.side_effect = [mock_tool1, mock_tool2]

        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=mock_output)
        mock_react_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, LocalDataValue)
        assert result.value == "ReAct agent execution result"

        assert mock_get_tool.call_count == 2
        mock_get_tool.assert_any_call(
            "search_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_get_tool.assert_any_call(
            "calculator_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )

        mock_react_agent_class.assert_called_once_with(
            task="Research and calculate",
            task_data=["context"],
            tools=[mock_tool1, mock_tool2],
            run_data=mock_run_data,
            tool_call_limit=30,
            allow_agent_clarifications=True,
            output_schema=None,
            model=mock_model,
        )

        mock_agent.execute.assert_called_once()


@pytest.mark.asyncio
async def test_react_agent_step_run_with_model() -> None:
    """Test ReActAgentStep run method with a specified model."""
    step = ReActAgentStep(
        task="Research", tools=["search_tool"], model="openai/gpt-4o", step_name="react_research"
    )
    mock_run_data = Mock()
    mock_tool = Mock()
    mock_output = LocalDataValue(value="result")

    with (
        patch("portia.builder.react_agent_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch("portia.builder.react_agent_step.ReActAgent") as mock_react_agent_class,
    ):
        mock_get_tool.return_value = mock_tool
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=mock_output)
        mock_react_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result == mock_output
        mock_react_agent_class.assert_called_once_with(
            task="Research",
            task_data=[],
            tools=[mock_tool],
            run_data=mock_run_data,
            tool_call_limit=25,
            allow_agent_clarifications=False,
            output_schema=None,
            model="openai/gpt-4o",
        )
        mock_agent.execute.assert_called_once()


@pytest.mark.asyncio
async def test_react_agent_step_run_with_reference_resolution() -> None:
    """Test ReActAgentStep run method with reference resolution in task and task_data."""
    ref1 = Input("user_query")
    ref2 = StepOutput(1)
    ref3 = Input("analysis_type")
    step = ReActAgentStep(
        task=f"Research and analyze {ref3} using {ref2}",
        tools=["search_tool"],
        step_name="react_research",
        inputs=["Static context", ref1, f"String interpolation: {ref2}"],
    )

    mock_run_data = Mock()
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="user_query", description="The user's query"),
        PlanInput(name="analysis_type", description="The type of analysis to perform"),
    ]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {
        "user_query": LocalDataValue(value="search query"),
        "analysis_type": LocalDataValue(value="sentiment analysis"),
    }
    mock_run_data.step_output_values = [
        Mock(
            value="analysis result",
            description="Analysis from step 1",
            step_name="analysis",
            step_num=1,
        )
    ]
    mock_run_data.plan.steps = []

    mock_output = LocalDataValue(value="ReAct execution with string interpolation")

    with (
        patch("portia.builder.react_agent_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch("portia.builder.react_agent_step.ReActAgent") as mock_react_agent_class,
    ):
        mock_tool = Mock()
        mock_get_tool.return_value = mock_tool

        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=mock_output)
        mock_react_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, LocalDataValue)
        assert result.value == "ReAct execution with string interpolation"

        mock_react_agent_class.assert_called_once()
        call_kwargs = mock_react_agent_class.call_args.kwargs

        resolved_task = call_kwargs["task"]
        assert resolved_task == "Research and analyze sentiment analysis using analysis result"
        task_data = call_kwargs["task_data"]
        assert len(task_data) == 3
        assert call_kwargs["model"] is None
        assert task_data[0] == "Static context"
        assert isinstance(task_data[1], LocalDataValue)
        assert task_data[1].value == "search query"
        assert task_data[1].summary == "The user's query"
        assert task_data[2] == "String interpolation: analysis result"
