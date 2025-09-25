"""Test the llm step."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.llm_step import LLMStep
from portia.builder.reference import Input, StepOutput
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput, Variable
from portia.plan import Step as PlanStep
from portia.run_context import StepOutputValue


class MockOutputSchema(BaseModel):
    """Mock output schema for testing."""

    result: str
    count: int


def test_llm_step_initialization() -> None:
    """Test LLMStep initialization."""
    step = LLMStep(task="Test task", step_name="llm_step")

    assert step.task == "Test task"
    assert step.step_name == "llm_step"
    assert step.inputs == []
    assert step.output_schema is None


def test_llm_step_initialization_with_all_parameters() -> None:
    """Test LLMStep initialization with all parameters."""
    inputs = [Input("user_query"), "additional context"]
    step = LLMStep(
        task="Analyze data",
        step_name="analysis",
        inputs=inputs,
        output_schema=MockOutputSchema,
    )

    assert step.task == "Analyze data"
    assert step.step_name == "analysis"
    assert step.inputs == inputs
    assert step.output_schema == MockOutputSchema


def test_llm_step_str() -> None:
    """Test LLMStep str method."""
    step = LLMStep(task="Test task", step_name="test")
    assert str(step) == "LLMStep(task='Test task')"


def test_llm_step_str_with_output_schema() -> None:
    """Test LLMStep str method with output schema."""
    step = LLMStep(task="Test task", step_name="test", output_schema=MockOutputSchema)
    assert str(step) == "LLMStep(task='Test task' -> MockOutputSchema)"


@pytest.mark.asyncio
async def test_llm_step_run_no_inputs() -> None:
    """Test LLMStep run with no inputs."""
    step = LLMStep(task="Analyze data", step_name="analysis")
    mock_run_data = Mock()
    mock_run_data.storage = Mock()

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Analysis complete")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Analysis complete"
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Analyze data"
        assert call_args[1]["task_data"] == []


@pytest.mark.asyncio
async def test_llm_step_run_one_regular_input() -> None:
    """Test LLMStep run with one regular value input."""
    step = LLMStep(task="Process text", step_name="process", inputs=["Hello world"])
    mock_run_data = Mock()
    mock_run_data.storage = Mock()

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Processed: Hello world")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Processed: Hello world"
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Process text"
        assert call_args[1]["task_data"] == ["Hello world"]


@pytest.mark.asyncio
async def test_llm_step_run_one_reference_input() -> None:
    """Test LLMStep run with one reference input."""
    reference_input = StepOutput(0)
    step = LLMStep(task="Summarize result", step_name="summarize", inputs=[reference_input])
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="previous step result",
            description="Step 0",
            step_name="summarize",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(reference_input, "get_value") as mock_get_value,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_get_value.return_value = "Previous step result"
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Summary: Previous step result")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Summary: Previous step result"
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Summarize result"
        expected_task_data = LocalDataValue(value="Previous step result", summary="Step 0")
        assert call_args[1]["task_data"] == [expected_task_data]


@pytest.mark.asyncio
async def test_llm_step_run_interpolated_reference_in_task() -> None:
    """Test LLMStep run where the task string contains an interpolated reference."""
    reference_input = StepOutput(0)
    step = LLMStep(
        task=f"Summarize result: {StepOutput(0)}",
        step_name="summarize",
        inputs=[reference_input],
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="previous step result",
            description="Step 0",
            step_name="summarize",
            step_num=0,
        )
    ]
    mock_run_data.plan.steps = []

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(reference_input, "get_value") as mock_get_value,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        # Only the input reference is patched; the task interpolation resolves
        # via StepOutput.get_value
        mock_get_value.return_value = "Previous step result"
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Summary: Previous step result")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Summary: Previous step result"
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Summarize result: previous step result"
        expected_task_data = LocalDataValue(value="Previous step result", summary="Step 0")
        assert call_args[1]["task_data"] == [expected_task_data]


@pytest.mark.asyncio
async def test_llm_step_run_mixed_inputs() -> None:
    """Test LLMStep run with 2 regular value inputs and 2 reference inputs."""
    ref1 = Input("user_name")
    ref2 = StepOutput(1)
    step = LLMStep(
        task="Generate report",
        step_name="report",
        inputs=["Context info", ref1, "Additional data", ref2],
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="Analysis complete",
            description="The output of step 1",
            step_name="summarize",
            step_num=1,
        )
    ]
    mock_run_data.plan = Mock()
    mock_run_data.plan.steps = []
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="user_name", description="The name of the user")
    ]

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(ref1, "get_value") as mock_get_value1,
        patch.object(ref2, "get_value") as mock_get_value2,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_get_value1.return_value = "John"
        mock_get_value2.return_value = "Analysis complete"
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Report generated successfully")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Report generated successfully"
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Generate report"

        expected_task_data = [
            "Context info",
            LocalDataValue(value="John", summary="The name of the user"),
            "Additional data",
            LocalDataValue(value="Analysis complete", summary="The output of step 1"),
        ]
        assert call_args[1]["task_data"] == expected_task_data


@pytest.mark.asyncio
async def test_llm_step_run_with_prompt() -> None:
    """Test LLMStep run with custom prompt parameter."""
    custom_prompt = "You are a helpful assistant. Please analyze the following data:"
    step = LLMStep(
        task="Analyze data",
        step_name="analysis",
        system_prompt=custom_prompt,
        output_schema=MockOutputSchema,
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Analysis complete")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Analysis complete"
        # Verify ToolCallWrapper was constructed and called properly
        mock_tool_wrapper_class.assert_called_once()
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Analyze data"
        assert call_args[1]["task_data"] == []


@pytest.mark.asyncio
async def test_llm_step_run_without_prompt() -> None:
    """Test LLMStep run without prompt parameter (uses default)."""
    step = LLMStep(
        task="Analyze data",
        step_name="analysis",
        output_schema=MockOutputSchema,
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Analysis complete")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Analysis complete"
        # Verify ToolCallWrapper was constructed and called properly
        mock_tool_wrapper_class.assert_called_once()
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Analyze data"
        assert call_args[1]["task_data"] == []


@pytest.mark.asyncio
async def test_llm_step_run_with_model() -> None:
    """Test LLMStep run with a specified model."""
    step = LLMStep(task="Analyze data", step_name="analysis", model="openai/gpt-4o")
    mock_run_data = Mock()
    mock_run_data.storage = Mock()

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch("portia.builder.llm_step.LLMTool") as mock_llm_tool_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_llm_tool_instance = Mock()
        mock_llm_tool_class.return_value = mock_llm_tool_instance
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Analysis complete")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Analysis complete"
        mock_llm_tool_class.assert_called_once_with(
            structured_output_schema=None, model="openai/gpt-4o"
        )
        mock_wrapper_instance.arun.assert_called_once()


@pytest.mark.asyncio
async def test_llm_step_run_with_string_template_input() -> None:
    """Test LLMStep run with an input string containing reference templates."""
    step = LLMStep(
        task="Summarize",
        step_name="summary",
        inputs=[f"Use {StepOutput(0)} and {Input('username')}"],
    )
    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="step0",
            description="s0",
            step_name="summary",
            step_num=0,
        )
    ]
    mock_run_data.plan = Mock()
    mock_run_data.plan.steps = []
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="done")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        mock_wrapper_instance.arun.assert_called_once()
        assert result == "done"
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task_data"] == ["Use step0 and Alice"]


def test_llm_step_to_legacy_step() -> None:
    """Test LLMStep to_legacy_step method."""
    inputs = [Input("user_query"), StepOutput(0)]
    step = LLMStep(
        task="Analyze data",
        step_name="analysis",
        inputs=inputs,
        output_schema=MockOutputSchema,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$analysis_output"

    # Mock the get_legacy_name method on the inputs
    with (
        patch.object(inputs[0], "get_legacy_name") as mock_input_name,
        patch.object(inputs[1], "get_legacy_name") as mock_stepoutput_name,
    ):
        mock_input_name.return_value = "user_query"
        mock_stepoutput_name.return_value = "step_0_output"

        legacy_step = step.to_legacy_step(mock_plan)

        # Verify the PlanStep has the correct attributes
        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Analyze data"
        assert legacy_step.tool_id == "llm_tool"  # LLMTool.LLM_TOOL_ID
        assert legacy_step.output == "$analysis_output"
        assert legacy_step.structured_output_schema == MockOutputSchema

        # Verify inputs are converted to Variables
        assert len(legacy_step.inputs) == 2
        assert all(isinstance(inp, Variable) for inp in legacy_step.inputs)
        assert legacy_step.inputs[0].name == "user_query"
        assert legacy_step.inputs[1].name == "step_0_output"

        # Verify mocks were called
        mock_input_name.assert_called_once_with(mock_plan)
        mock_stepoutput_name.assert_called_once_with(mock_plan)
        mock_plan.step_output_name.assert_called_once_with(step)


@pytest.mark.asyncio
async def test_llm_step_run_linked_inputs() -> None:
    """Test LLMStep run with 2 inputs, one that refers to a Step Output."""
    ref1 = Input("user_name")
    ref2 = Input("user_height")
    step = LLMStep(
        task="Generate report",
        step_name="report",
        inputs=["Context info", ref1, "Additional data", ref2],
    )
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="Analysis complete",
            description="The output of step 1",
            step_name="summarize",
            step_num=1,
        )
    ]
    mock_run_data.plan = Mock()
    mock_run_data.plan.steps = []
    # A plan input being the output value from another step can happen with linked plans using
    # .add_sub_plan().
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="user_name"),
        PlanInput(name="user_height", value=StepOutput(1)),
    ]

    with (
        patch("portia.builder.llm_step.ToolCallWrapper") as mock_tool_wrapper_class,
        patch.object(mock_run_data, "get_tool_run_ctx") as mock_get_tool_run_ctx,
        patch.object(ref1, "get_value") as mock_get_value1,
        patch.object(ref2, "get_value") as mock_get_value2,
    ):
        mock_get_tool_run_ctx.return_value = Mock()
        mock_get_value1.return_value = "John"
        mock_get_value2.return_value = "6ft"
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.arun = AsyncMock(return_value="Report generated successfully")
        mock_tool_wrapper_class.return_value = mock_wrapper_instance

        result = await step.run(run_data=mock_run_data)

        assert result == "Report generated successfully"
        mock_wrapper_instance.arun.assert_called_once()
        call_args = mock_wrapper_instance.arun.call_args
        assert call_args[1]["task"] == "Generate report"

        expected_task_data = [
            "Context info",
            LocalDataValue(value="John", summary=""),
            "Additional data",
            LocalDataValue(value="6ft", summary="The output of step 1"),
        ]
        assert call_args[1]["task_data"] == expected_task_data
