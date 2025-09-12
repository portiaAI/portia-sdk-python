"""Test the invoke tool step."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.invoke_tool_step import InvokeToolStep
from portia.builder.reference import Input, StepOutput
from portia.clarification import (
    Clarification,
    ClarificationCategory,
)
from portia.errors import ToolNotFoundError
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.plan import Step as PlanStep
from portia.prefixed_uuid import PlanRunUUID
from portia.run_context import StepOutputValue
from portia.tool import Tool
from portia.tool_decorator import tool


class MockOutputSchema(BaseModel):
    """Mock output schema for testing."""

    result: str
    count: int


def example_function(x: int, y: str) -> str:
    """Return formatted string for testing FunctionStep."""
    return f"{y}: {x}"


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


def test_invoke_tool_step_initialization_with_string_tool() -> None:
    """Test InvokeToolStep initialization with string tool."""
    step = InvokeToolStep(tool="search_tool", step_name="search")

    assert step.tool == "search_tool"
    assert step.step_name == "search"
    assert step.args == {}
    assert step.output_schema is None


def test_invoke_tool_step_initialization_with_tool_instance() -> None:
    """Test InvokeToolStep initialization with Tool instance."""
    mock_tool = MockTool()
    args = {"query": "test", "limit": StepOutput(0)}

    step = InvokeToolStep(
        tool=mock_tool,
        step_name="search",
        args=args,
        output_schema=MockOutputSchema,
    )

    assert step.tool is mock_tool
    assert step.args == args
    assert step.output_schema == MockOutputSchema


def test_invoke_tool_step_str_with_string_tool() -> None:
    """Test InvokeToolStep str method with string tool."""
    step = InvokeToolStep(
        tool="search_tool",
        step_name="search",
        args={"query": "test"},
    )

    assert str(step) == "InvokeToolStep(tool='search_tool', args={'query': 'test'})"


def test_invoke_tool_step_str_with_tool_instance() -> None:
    """Test InvokeToolStep str method with Tool instance."""
    mock_tool = MockTool()
    step = InvokeToolStep(
        tool=mock_tool,
        step_name="search",
        args={"query": "test"},
        output_schema=MockOutputSchema,
    )

    assert (
        str(step) == "InvokeToolStep(tool='mock_tool', args={'query': 'test'} -> MockOutputSchema)"
    )


@pytest.mark.asyncio
async def test_invoke_tool_step_with_regular_value_input() -> None:
    """Test InvokeToolStep run with 1 regular value input."""
    step = InvokeToolStep(tool="mock_tool", step_name="run_tool", args={"query": "search term"})
    mock_run_data = Mock()
    mock_tool = Mock()
    mock_tool.structured_output_schema = None
    mock_output = Mock()
    mock_output.get_value.return_value = "tool result"
    mock_tool._arun = AsyncMock(return_value=mock_output)

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert result == "tool result"
        mock_get_tool.assert_called_once_with(
            "mock_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_tool._arun.assert_called_once_with(mock_tool_ctx, query="search term")


@pytest.mark.asyncio
async def test_invoke_tool_step_with_regular_value_input_and_output_schema() -> None:
    """Test InvokeToolStep run with 1 regular value input and output schema."""
    step = InvokeToolStep(
        tool="mock_tool",
        step_name="run_tool",
        args={"query": "search term"},
        output_schema=MockOutputSchema,
    )
    mock_run_data = Mock()
    mock_tool = Mock()
    mock_output = Mock()
    mock_output.get_value.return_value = "raw tool result"
    mock_tool._arun = AsyncMock(return_value=mock_output)

    # Mock the model and its aget_structured_response method
    mock_model = Mock()
    mock_model.aget_structured_response = AsyncMock(
        return_value=MockOutputSchema(result="structured result", count=5)
    )

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(mock_run_data.config, "get_default_model") as mock_get_model,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_get_model.return_value = mock_model
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, MockOutputSchema)
        assert result.result == "structured result"
        mock_get_tool.assert_called_once_with(
            "mock_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_tool._arun.assert_called_once_with(mock_tool_ctx, query="search term")
        mock_model.aget_structured_response.assert_called_once()


@pytest.mark.asyncio
async def test_invoke_tool_step_with_tool_output_schema() -> None:
    """Test InvokeToolStep run with 1 regular value input and output schema."""
    step = InvokeToolStep(
        tool="mock_tool",
        step_name="run_tool",
    )
    mock_run_data = Mock()
    mock_tool = Mock()
    mock_tool.structured_output_schema = MockOutputSchema
    mock_output = Mock()
    mock_output.get_value.return_value = "raw tool result"
    mock_tool._arun = AsyncMock(return_value=mock_output)

    # Mock the model and its aget_structured_response method
    mock_model = Mock()
    mock_model.aget_structured_response = AsyncMock(
        return_value=MockOutputSchema(result="structured result", count=5)
    )

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(mock_run_data.config, "get_default_model") as mock_get_model,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_get_model.return_value = mock_model
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, MockOutputSchema)
        assert result.result == "structured result"
        mock_get_tool.assert_called_once_with(
            "mock_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_tool._arun.assert_called_once_with(mock_tool_ctx)
        mock_model.aget_structured_response.assert_called_once()


@pytest.mark.asyncio
async def test_invoke_tool_step_with_reference_input() -> None:
    """Test InvokeToolStep run with 1 reference input."""
    reference_input = StepOutput(0)
    step = InvokeToolStep(tool="mock_tool", step_name="run_tool", args={"query": reference_input})
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_tool = Mock()
    mock_tool.structured_output_schema = None
    mock_output = Mock()
    mock_output.get_value.return_value = "tool result with reference"
    mock_tool._arun = AsyncMock(return_value=mock_output)

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(reference_input, "get_value") as mock_get_value,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_get_value.return_value = "previous step output"
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert result == "tool result with reference"
        mock_get_tool.assert_called_once_with(
            "mock_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_tool._arun.assert_called_once_with(mock_tool_ctx, query="previous step output")


@pytest.mark.asyncio
async def test_invoke_tool_step_with_mixed_inputs() -> None:
    """Test InvokeToolStep run with 2 regular value inputs and 2 reference inputs."""
    ref1 = Input("user_query")
    ref2 = StepOutput(1)
    step = InvokeToolStep(
        tool="mock_tool",
        step_name="run_tool",
        args={
            "context": "static context",
            "user_input": ref1,
            "limit": 10,
            "previous_result": ref2,
        },
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_tool = Mock()
    mock_tool.structured_output_schema = None
    mock_output = Mock()
    mock_output.get_value.return_value = "mixed inputs result"
    mock_tool._arun = AsyncMock(return_value=mock_output)

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(ref1, "get_value") as mock_get_value1,
        patch.object(ref2, "get_value") as mock_get_value2,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_get_value1.return_value = "user question"
        mock_get_value2.return_value = "step 1 output"
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert result == "mixed inputs result"
        mock_get_tool.assert_called_once_with(
            "mock_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_tool._arun.assert_called_once_with(
            mock_tool_ctx,
            context="static context",
            user_input="user question",
            limit=10,
            previous_result="step 1 output",
        )


@pytest.mark.asyncio
async def test_invoke_tool_step_no_args_with_clarification() -> None:
    """Test InvokeToolStep run with no args and tool returns a clarification."""
    step = InvokeToolStep(tool="mock_tool", step_name="run_tool", args={})
    mock_run_data = Mock()
    mock_tool = Mock()
    mock_tool.structured_output_schema = None
    mock_clarification = Clarification(
        category=ClarificationCategory.ACTION,
        user_guidance="Need more information",
        plan_run_id=None,
    )
    mock_output = Mock()
    mock_output.get_value.return_value = mock_clarification
    mock_tool._arun = AsyncMock(return_value=mock_output)

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, Clarification)
        assert result.user_guidance == "Need more information"
        assert result.plan_run_id == mock_run_data.plan_run.id
        mock_get_tool.assert_called_once_with(
            "mock_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )
        mock_tool._arun.assert_called_once_with(mock_tool_ctx)


@pytest.mark.asyncio
async def test_invoke_tool_step_with_tool_instance() -> None:
    """Test InvokeToolStep run with Tool instance instead of string tool."""
    mock_tool = MockTool()
    step = InvokeToolStep(tool=mock_tool, step_name="run_tool", args={"input": "test input"})
    mock_run_data = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.current_step_index = 0
    mock_run_data.storage = AsyncMock()

    with (
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(mock_tool, "_arun") as mock_arun,
    ):
        mock_ctx = Mock()
        mock_ctx.end_user.external_id = "test_user_id"
        mock_get_tool_run_ctx.return_value = mock_ctx
        mock_output = Mock()
        mock_output.get_value.return_value = "mock result"
        mock_arun.return_value = mock_output

        result = await step.run(run_data=mock_run_data)

        assert result == "mock result"
        mock_arun.assert_called_once_with(mock_ctx, input="test input")


@pytest.mark.asyncio
async def test_invoke_tool_step_with_nonexistent_tool_id() -> None:
    """Test InvokeToolStep run with nonexistent tool_id raises ToolNotFoundError."""
    step = InvokeToolStep(tool="nonexistent_tool", step_name="run_tool", args={"query": "test"})
    mock_run_data = Mock()

    with patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool:
        mock_get_tool.return_value = None  # Tool not found

        with pytest.raises(ToolNotFoundError) as exc_info:
            await step.run(run_data=mock_run_data)

        assert "nonexistent_tool" in str(exc_info.value)
        mock_get_tool.assert_called_once_with(
            "nonexistent_tool",
            mock_run_data.tool_registry,
            mock_run_data.storage,
            mock_run_data.plan_run,
        )


@pytest.mark.asyncio
async def test_invoke_tool_step_with_function_tool() -> None:
    """Test InvokeToolStep run with nonexistent tool_id raises ToolNotFoundError."""
    tool_class = tool(example_function)
    step = InvokeToolStep(tool=tool_class(), step_name="run_tool", args={"x": 42, "y": "Result"})
    mock_run_data = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.current_step_index = 0
    mock_run_data.storage = AsyncMock()

    with patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx:
        mock_ctx = Mock()
        mock_ctx.end_user.external_id = "test_user_id"
        mock_get_tool_run_ctx.return_value = mock_ctx
        result = await step.run(run_data=mock_run_data)

        assert result == "Result: 42"


@pytest.mark.asyncio
async def test_invoke_tool_step_with_async_function_tool() -> None:
    """Test InvokeToolStep run with nonexistent tool_id raises ToolNotFoundError."""

    async def async_example_function(x: int, y: str) -> str:
        await asyncio.sleep(0.001)
        return f"{y}: {x}"

    tool_class = tool(async_example_function)
    step = InvokeToolStep(tool=tool_class(), step_name="run_tool", args={"x": 42, "y": "Result"})
    mock_run_data = Mock()
    # Configure mock to return proper values for ToolCallRecord
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.current_step_index = 0
    mock_run_data.storage = AsyncMock()

    with patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx:
        mock_ctx = Mock()
        mock_ctx.end_user.external_id = "test_user_id"
        mock_get_tool_run_ctx.return_value = mock_ctx
        result = await step.run(run_data=mock_run_data)

        assert result == "Result: 42"


@pytest.mark.asyncio
async def test_invoke_tool_step_with_string_arg_templates() -> None:
    """Test InvokeToolStep run with string args containing reference templates."""
    step = InvokeToolStep(
        tool="mock_tool",
        step_name="run_tool",
        args={"query": f"Search {StepOutput(0)} for {Input('username')}"},
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="result",
            description="s0",
            step_name="run_tool",
            step_num=0,
        )
    ]
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}

    mock_tool = Mock()
    mock_tool.structured_output_schema = None
    mock_output = Mock()
    mock_output.get_value.return_value = "final"
    mock_tool._arun = AsyncMock(return_value=mock_output)

    with (
        patch("portia.builder.invoke_tool_step.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool.return_value = mock_tool
        mock_tool_ctx = Mock()
        mock_get_tool_run_ctx.return_value = mock_tool_ctx

        result = await step.run(run_data=mock_run_data)

        assert result == "final"
        mock_tool._arun.assert_called_once_with(mock_tool_ctx, query="Search result for Alice")


def test_invoke_tool_step_to_legacy_step() -> None:
    """Test InvokeToolStep to_legacy_step method."""
    args = {"query": Input("user_query"), "limit": 10}
    step = InvokeToolStep(
        tool="search_tool",
        step_name="search",
        args=args,
        output_schema=MockOutputSchema,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$search_output"

    with patch.object(args["query"], "get_legacy_name") as mock_input_name:
        mock_input_name.return_value = "user_query"

        legacy_step = step.to_legacy_step(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Use tool search_tool with args: query=$user_query, limit=10"
        assert legacy_step.tool_id == "search_tool"
        assert legacy_step.output == "$search_output"
        assert legacy_step.structured_output_schema == MockOutputSchema

        assert len(legacy_step.inputs) == 1
        assert legacy_step.inputs[0].name == "user_query"
