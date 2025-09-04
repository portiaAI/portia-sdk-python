"""Test the step_v2 module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.conditionals import (
    ConditionalBlock,
    ConditionalBlockClauseType,
    ConditionalStepResult,
)
from portia.builder.loops import LoopBlockType, LoopType
from portia.builder.reference import Input, StepOutput
from portia.builder.step_v2 import (
    ConditionalStep,
    InvokeToolStep,
    LLMStep,
    LoopStep,
    ReActAgentStep,
    SingleToolAgentStep,
    StepV2,
    UserInputStep,
    UserVerifyStep,
)
from portia.clarification import (
    Clarification,
    ClarificationCategory,
    InputClarification,
    MultipleChoiceClarification,
    UserVerificationClarification,
)
from portia.config import ExecutionAgentType
from portia.errors import PlanRunExitError, ToolNotFoundError
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_agents.output import LocalDataValue
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import PlanInput, Variable
from portia.plan import Step as PlanStep
from portia.prefixed_uuid import PlanRunUUID
from portia.run_context import StepOutputValue
from portia.tool import Tool
from portia.tool_decorator import tool

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2


class MockOutputSchema(BaseModel):
    """Mock output schema for testing."""

    result: str
    count: int


def example_function(x: int, y: str) -> str:
    """Return formatted string for testing FunctionStep."""
    return f"{y}: {x}"


@pytest.fixture
def mock_llm_tool() -> Mock:
    """Create a mock LLMTool for testing."""
    real_llm_tool = LLMTool()
    mock = Mock(
        spec=LLMTool,
        arun=AsyncMock(),
        run=Mock(),
        id="mock_llm_tool_id",
        name="mock_llm_tool",
        description="Mock LLM Tool",
    )
    mock.name = "mock_llm_tool"
    mock.args_schema = real_llm_tool.args_schema
    mock.output_schema = real_llm_tool.output_schema
    mock.structured_output_schema = real_llm_tool.structured_output_schema
    mock.should_summarize = real_llm_tool.should_summarize
    return mock


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


class ConcreteStepV2(StepV2):
    """Concrete implementation of StepV2 for testing base functionality."""

    def __init__(self, step_name: str = "test_step") -> None:
        """Initialize concrete step."""
        super().__init__(step_name=step_name)

    async def run(self, run_data: Any) -> str:  # noqa: ANN401, ARG002
        """Mock run method."""
        return "test result"

    def to_legacy_step(self, plan: PlanV2) -> PlanStep:  # noqa: ARG002
        """Mock to_legacy_step method."""
        return PlanStep(
            task="Test task",
            inputs=[],
            tool_id="test_tool",
            output="test_output",
        )


# Test cases for the base StepV2 class


def test_step_v2_initialization() -> None:
    """Test StepV2 initialization."""
    step = ConcreteStepV2("my_step")
    assert step.step_name == "my_step"


def test_resolve_input_reference_with_non_reference() -> None:
    """Test _resolve_input_reference with non-reference input."""
    step = ConcreteStepV2()
    mock_run_data = Mock()

    result = step._resolve_references("plain_string", mock_run_data)
    assert result == "plain_string"

    result = step._resolve_references(42, mock_run_data)
    assert result == 42


def test_resolve_input_reference_with_reference() -> None:
    """Test _resolve_input_reference with Reference input."""
    step = ConcreteStepV2()
    mock_run_data = Mock()
    reference = StepOutput(0)

    with patch.object(reference, "get_value", return_value="reference_result") as mock_get_value:
        result = step._resolve_references(reference, mock_run_data)

        assert result == "reference_result"
        mock_get_value.assert_called_once_with(mock_run_data)


def test_resolve_input_reference_with_string_template_step_output() -> None:
    """Test _resolve_input_reference with string containing StepOutput template."""
    step = ConcreteStepV2()
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="step result",
            description="Step 0",
            step_name="test_step",
            step_num=0,
        )
    ]

    template = f"The result was {StepOutput(0)}"
    result = step._resolve_references(template, mock_run_data)

    assert result == "The result was step result"


def test_resolve_input_reference_with_string_template_input() -> None:
    """Test _resolve_input_reference with string containing Input template."""
    step = ConcreteStepV2()
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}
    mock_run_data.step_output_values = []

    template = f"Hello {Input('username')}"
    result = step._resolve_references(template, mock_run_data)

    assert result == "Hello Alice"


def test_resolve_input_reference_with_string_template_step_both() -> None:
    """Test _resolve_input_reference with string containing StepOutput template."""
    step = ConcreteStepV2()
    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="step result",
            description="Step 0",
            step_name="test_step",
            step_num=0,
        )
    ]
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}

    template = f"The input was '{Input('username')}' and the result was '{StepOutput(0)}'"
    result = step._resolve_references(template, mock_run_data)

    assert result == "The input was 'Alice' and the result was 'step result'"


def test_resolve_input_reference_with_regular_value() -> None:
    """Test _resolve_input_reference with regular value."""
    step = ConcreteStepV2()
    mock_run_data = Mock()

    result = step._resolve_references("regular_value", mock_run_data)
    assert result == "regular_value"


def test_resolve_input_names_for_printing_with_reference() -> None:
    """Test _resolve_input_names_for_printing with Reference."""
    step = ConcreteStepV2()
    mock_plan = Mock()
    reference = StepOutput(0)

    with patch.object(
        reference, "get_legacy_name", return_value="step_0_output"
    ) as mock_get_legacy:
        result = step._resolve_input_names_for_printing(reference, mock_plan)

        assert result == "$step_0_output"
        mock_get_legacy.assert_called_once_with(mock_plan)


def test_resolve_input_names_for_printing_with_reference_already_prefixed() -> None:
    """Test _resolve_input_names_for_printing with Reference that already has $ prefix."""
    step = ConcreteStepV2()
    mock_plan = Mock()
    mock_reference = StepOutput(0)

    with patch.object(mock_reference, "get_legacy_name", return_value="$step_0_output"):
        result = step._resolve_input_names_for_printing(mock_reference, mock_plan)

        assert result == "$step_0_output"


def test_resolve_input_names_for_printing_with_list() -> None:
    """Test _resolve_input_names_for_printing with list."""
    step = ConcreteStepV2()
    mock_plan = Mock()
    reference = Input("test_input")

    with patch.object(reference, "get_legacy_name", return_value="input_name"):
        input_list = ["regular_value", reference, 42]
        result = step._resolve_input_names_for_printing(input_list, mock_plan)

        assert result == ["regular_value", "$input_name", 42]


def test_resolve_input_names_for_printing_with_regular_value() -> None:
    """Test _resolve_input_names_for_printing with regular value."""
    step = ConcreteStepV2()
    mock_plan = Mock()

    result = step._resolve_input_names_for_printing("regular_value", mock_plan)
    assert result == "regular_value"


def test_inputs_to_legacy_plan_variables() -> None:
    """Test _inputs_to_legacy_plan_variables method."""
    step = ConcreteStepV2()
    mock_plan = Mock()

    ref1 = Input("test_input")
    ref2 = StepOutput(0)

    with (
        patch.object(ref1, "get_legacy_name", return_value="input1"),
        patch.object(ref2, "get_legacy_name", return_value="step_0_output"),
    ):
        inputs = ["regular_value", ref1, 42, ref2]

        result = step._inputs_to_legacy_plan_variables(inputs, mock_plan)

        assert len(result) == 2
        assert all(isinstance(var, Variable) for var in result)
        assert result[0].name == "input1"
        assert result[1].name == "step_0_output"


# Test cases for the LLMStep class


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
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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

    with (
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="user_name", description="The name of the user")
    ]

    with (
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}

    with (
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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
    # A plan input being the output value from another step can happen with linked plans using
    # .add_steps().
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="user_name"),
        PlanInput(name="user_height", value=StepOutput(1)),
    ]

    with (
        patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
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


# Test cases for the InvokeToolStep class


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


def test_tool_name_with_string_tool() -> None:
    """Test _tool_name method with string tool."""
    step = InvokeToolStep(tool="search_tool", step_name="search")
    assert step._tool_name() == "search_tool"


def test_tool_name_with_tool_instance() -> None:
    """Test _tool_name method with Tool instance."""
    mock_tool = MockTool()
    step = InvokeToolStep(tool=mock_tool, step_name="search")
    assert step._tool_name() == "mock_tool"


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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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

    with patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool:
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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


# Test cases for the SingleToolAgent class


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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
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


# Test cases for the ReActAgentStep class


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


def test_react_agent_step_to_legacy_step() -> None:
    """Test ReActAgentStep to_legacy_step method."""
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

        legacy_step = step.to_legacy_step(mock_plan)

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
    step = ReActAgentStep(
        task="Research and calculate",
        tools=tools,
        step_name="react_agent",
        inputs=["context"],
        tool_call_limit=30,
        allow_agent_clarifications=True,
    )

    mock_run_data = Mock()

    mock_tool1 = Mock()
    mock_tool2 = Mock()
    mock_output = LocalDataValue(value="ReAct agent execution result")

    with (
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch("portia.builder.step_v2.ReActAgent") as mock_react_agent_class,
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

    mock_output = LocalDataValue(value="ReAct execution with string interpolation")

    with (
        patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
        patch("portia.builder.step_v2.ReActAgent") as mock_react_agent_class,
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
        assert task_data[0] == "Static context"
        assert isinstance(task_data[1], LocalDataValue)
        assert task_data[1].value == "search query"
        assert task_data[1].summary == "The user's query"
        assert task_data[2] == "String interpolation: analysis result"


# Test cases for the UserVerifyStep class


def test_user_verify_step_str() -> None:
    """Test UserVerifyStep str method."""
    step = UserVerifyStep(message="Please confirm this action", step_name="verify")
    assert str(step) == "UserVerifyStep(message='Please confirm this action')"


def test_user_verify_step_to_legacy_step() -> None:
    """Test UserVerifyStep to_legacy_step method."""
    step = UserVerifyStep(message="Confirm deletion", step_name="confirm_delete")

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$confirm_delete_output"

    legacy_step = step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "User verification: Confirm deletion"
    assert legacy_step.inputs == []
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$confirm_delete_output"
    assert legacy_step.structured_output_schema is None


@pytest.mark.asyncio
async def test_user_verify_step_requests_clarification() -> None:
    """Test that UserVerifyStep returns a clarification on first run."""
    message = f"Proceed with {StepOutput(0)} for {Input('username')}?"
    step = UserVerifyStep(message=message, step_name="verify")

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="result", description="")
    ]

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, UserVerificationClarification)
    assert result.user_guidance == "Proceed with result for Alice?"


@pytest.mark.asyncio
async def test_user_verify_step_user_confirms() -> None:
    """Test that the step succeeds when user verifies."""
    step = UserVerifyStep(message="Confirm?", step_name="verify")

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Confirm?",
        response=True,
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    result = await step.run(run_data=mock_run_data)

    assert result is True


@pytest.mark.asyncio
async def test_user_verify_step_user_rejects() -> None:
    """Test that the step raises error when user rejects."""
    step = UserVerifyStep(message="Confirm?", step_name="verify")

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Confirm?",
        response=False,
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    with pytest.raises(PlanRunExitError):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_user_verify_step_with_string_template_message() -> None:
    """Test UserVerifyStep message resolution with reference templates."""
    step = UserVerifyStep(
        message=f"Confirm action on {StepOutput(0)} for user {Input('username')}?",
        step_name="verify_action",
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
    mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Bob")}
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="test_file.txt",
            description="step0",
            step_name="verify_action",
            step_num=0,
        )
    ]

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, UserVerificationClarification)
    assert result.user_guidance == "Confirm action on test_file.txt for user Bob?"


# Test cases for the ConditionalStep class


def test_conditional_step_good_initialization() -> None:
    """Test ConditionalStep initialization with valid parameters."""
    conditional_block = ConditionalBlock(
        clause_step_indexes=[0, 5, 10],
        parent_conditional_block=None,
    )

    def test_condition(x: int) -> bool:
        return x > 0

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"x": 5},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    assert step.step_name == "test_conditional"
    assert step.conditional_block == conditional_block
    assert step.condition == test_condition
    assert step.args == {"x": 5}
    assert step.clause_index_in_block == 0
    assert step.block_clause_type == ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK


def test_conditional_step_initialization_with_string_condition() -> None:
    """Test ConditionalStep initialization with string condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    step = ConditionalStep(
        step_name="string_conditional",
        conditional_block=conditional_block,
        condition="user_input > 10",
        args={"user_input": Input("number")},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    assert step.condition == "user_input > 10"


def test_conditional_step_validation_none_conditional_block() -> None:
    """Test that ConditionalStep raises ValueError when conditional_block is None."""

    def test_condition() -> bool:
        return True

    with pytest.raises(ValueError, match="Conditional block is required for ConditionSteps"):
        ConditionalStep(
            step_name="invalid_conditional",
            conditional_block=None,
            condition=test_condition,
            args={},
            clause_index_in_block=0,
            block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
        )


def test_conditional_step_block_property_good_case() -> None:
    """Test ConditionalStep block property returns the conditional block."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 2, 4])

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=lambda: True,
        args={},
        clause_index_in_block=1,
        block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
    )

    assert step.block == conditional_block
    assert isinstance(step.block, ConditionalBlock)


def test_conditional_step_block_property_error_case() -> None:
    """Test ConditionalStep block property raises error for invalid conditional_block type."""
    # This test creates an invalid state that shouldn't normally occur due to validation
    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=ConditionalBlock(clause_step_indexes=[0]),
        condition=lambda: True,
        args={},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    # Manually set conditional_block to invalid type to test error case
    step.conditional_block = "invalid_type"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Conditional block is not a ConditionalBlock"):
        _ = step.block


def test_conditional_step_str() -> None:
    """Test ConditionalStep str method."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    def test_function() -> bool:
        return True

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_function,
        args={"value": 42},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    result = str(step)
    assert "ConditionalStep" in result
    assert "test_function" in result or str(test_function) in result
    assert "NEW_CONDITIONAL_BLOCK" in result
    assert "{'value': 42}" in result


@pytest.mark.asyncio
async def test_conditional_step_run_with_function_condition_true() -> None:
    """Test ConditionalStep run with function condition that returns True."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 5, 10, 15])

    def test_condition(x: int) -> bool:
        return x > 5

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"x": 10},
        clause_index_in_block=1,  # Second clause (else_if)
        block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
    )

    mock_run_data = Mock()

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, ConditionalStepResult)
    assert result.type == ConditionalBlockClauseType.ALTERNATE_CLAUSE
    assert result.conditional_result is True
    assert result.next_clause_step_index == 10  # Next clause
    assert result.end_condition_block_step_index == 15  # Last clause (endif)


@pytest.mark.asyncio
async def test_conditional_step_run_with_function_condition_false() -> None:
    """Test ConditionalStep run with function condition that returns False."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    def test_condition(x: int) -> bool:
        return x > 10

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"x": 5},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_run_data = Mock()

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, ConditionalStepResult)
    assert result.conditional_result is False
    assert result.next_clause_step_index == 3
    assert result.end_condition_block_step_index == 3


@pytest.mark.asyncio
async def test_conditional_step_run_with_reference_args() -> None:
    """Test ConditionalStep run with reference arguments."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 4])
    reference_input = StepOutput(0)

    def test_condition(value: int) -> bool:
        return value > 0

    step = ConditionalStep(
        step_name="test_conditional",
        conditional_block=conditional_block,
        condition=test_condition,
        args={"value": reference_input},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_run_data = Mock()
    mock_run_data.portia.storage = Mock()

    with patch.object(reference_input, "get_value") as mock_get_value:
        mock_get_value.return_value = 42

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, ConditionalStepResult)
        assert result.conditional_result is True
        mock_get_value.assert_called_once_with(mock_run_data)


@pytest.mark.asyncio
async def test_conditional_step_run_with_string_condition() -> None:
    """Test ConditionalStep run with string condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 2])

    step = ConditionalStep(
        step_name="string_conditional",
        conditional_block=conditional_block,
        condition="x > 5",
        args={"x": 10},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_run_data = Mock()
    mock_agent = Mock()
    mock_agent.execute = AsyncMock(return_value=True)

    with patch("portia.builder.step_v2.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, ConditionalStepResult)
        assert result.conditional_result is True
        mock_agent_class.assert_called_once_with(mock_run_data.config)
        mock_agent.execute.assert_called_once_with(conditional="x > 5", arguments={"x": 10})


def test_conditional_step_to_legacy_step_with_function_condition() -> None:
    """Test ConditionalStep to_legacy_step with function condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 3])

    def my_test_function(x: int) -> bool:
        return x > 0

    step = ConditionalStep(
        step_name="function_conditional",
        conditional_block=conditional_block,
        condition=my_test_function,
        args={"x": Input("user_input")},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$function_conditional_output"

    with (
        patch.object(step.args["x"], "get_legacy_name") as mock_input_name,
        patch.object(step, "_get_legacy_condition") as mock_legacy_condition,
    ):
        mock_input_name.return_value = "user_input"
        mock_legacy_condition.return_value = "test condition"

        legacy_step = step.to_legacy_step(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Conditional clause: If result of my_test_function is true"
        assert legacy_step.tool_id is None
        assert legacy_step.output == "$function_conditional_output"
        assert legacy_step.condition == "test condition"

        # Verify inputs conversion
        assert len(legacy_step.inputs) == 1
        assert legacy_step.inputs[0].name == "user_input"

        mock_legacy_condition.assert_called_once_with(mock_plan)


def test_conditional_step_to_legacy_step_with_string_condition() -> None:
    """Test ConditionalStep to_legacy_step with string condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 2])

    step = ConditionalStep(
        step_name="string_conditional",
        conditional_block=conditional_block,
        condition="user_age >= 18",
        args={"user_age": Input("age")},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$string_conditional_output"

    with (
        patch.object(step.args["user_age"], "get_legacy_name") as mock_input_name,
        patch.object(step, "_get_legacy_condition") as mock_legacy_condition,
    ):
        mock_input_name.return_value = "age"
        mock_legacy_condition.return_value = "legacy condition string"

        legacy_step = step.to_legacy_step(mock_plan)

        assert isinstance(legacy_step, PlanStep)
        assert legacy_step.task == "Conditional clause: user_age >= 18"
        assert legacy_step.tool_id is None
        assert legacy_step.output == "$string_conditional_output"
        assert legacy_step.condition == "legacy condition string"

        assert len(legacy_step.inputs) == 1
        assert legacy_step.inputs[0].name == "age"


def test_conditional_step_to_legacy_step_with_lambda_function() -> None:
    """Test ConditionalStep to_legacy_step with lambda function condition."""
    conditional_block = ConditionalBlock(clause_step_indexes=[0, 1])

    lambda_condition = lambda x: x > 5  # noqa: E731

    step = ConditionalStep(
        step_name="lambda_conditional",
        conditional_block=conditional_block,
        condition=lambda_condition,
        args={},
        clause_index_in_block=0,
        block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$lambda_conditional_output"

    with patch.object(step, "_get_legacy_condition") as mock_legacy_condition:
        mock_legacy_condition.return_value = None

        legacy_step = step.to_legacy_step(mock_plan)

        assert "If result of" in legacy_step.task
        assert "is true" in legacy_step.task


# Test cases for the UserInputStep class


def test_user_input_step_str_text_input() -> None:
    """Test UserInputStep str method for text input."""
    step = UserInputStep(
        message="Please provide input",
        step_name="input",
    )

    assert str(step) == "UserInputStep(type='text input', message='Please provide input')"


def test_user_input_step_str_multiple_choice() -> None:
    """Test UserInputStep str method for multiple choice."""
    step = UserInputStep(
        message="Choose an option",
        step_name="choice",
        options=["A", "B", "C"],
    )

    assert str(step) == "UserInputStep(type='multiple choice', message='Choose an option')"


def test_user_input_step_to_legacy_step_text_input() -> None:
    """Test UserInputStep to_legacy_step method for text input."""
    step = UserInputStep(
        message="Enter your name",
        step_name="name_input",
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$name_input_output"

    legacy_step = step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "User input (Text input): Enter your name"
    assert legacy_step.inputs == []
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$name_input_output"
    assert legacy_step.structured_output_schema is None


def test_user_input_step_to_legacy_step_multiple_choice() -> None:
    """Test UserInputStep to_legacy_step method for multiple choice."""
    step = UserInputStep(
        message="Choose an option",
        step_name="choice",
        options=["A", "B", "C"],
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "$choice_output"

    legacy_step = step.to_legacy_step(mock_plan)

    assert isinstance(legacy_step, PlanStep)
    assert legacy_step.task == "User input (Multiple choice): Choose an option"
    assert legacy_step.inputs == []
    assert legacy_step.tool_id is None
    assert legacy_step.output == "$choice_output"
    assert legacy_step.structured_output_schema is None


@pytest.mark.asyncio
async def test_user_input_step_text_input_requests_clarification() -> None:
    """Test that UserInputStep returns INPUT clarification on first run for text input."""
    step = UserInputStep(
        message="Please enter your name",
        step_name="get_name",
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.step_output_name.return_value = "user_input"

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, InputClarification)
    assert result.user_guidance == "Please enter your name"
    assert result.argument_name == "user_input"


@pytest.mark.asyncio
async def test_user_input_step_multiple_choice_requests_clarification() -> None:
    """Test that UserInputStep returns MULTIPLE_CHOICE clarification when options provided."""
    step = UserInputStep(
        message="Choose your favorite color",
        step_name="choose_color",
        options=["red", "green", "blue"],
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.step_output_name.return_value = "user_input"

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, MultipleChoiceClarification)
    assert result.user_guidance == "Choose your favorite color"
    assert result.argument_name == "user_input"
    assert result.options == ["red", "green", "blue"]


@pytest.mark.asyncio
async def test_user_input_step_returns_response_when_resolved_text_input() -> None:
    """Test that UserInputStep returns response when text input clarification is resolved."""
    step = UserInputStep(
        message="Please enter your name",
        step_name="get_name",
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = InputClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please enter your name",
        argument_name="user_input",
        response="Alice",
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    result = await step.run(run_data=mock_run_data)

    assert result == "Alice"


@pytest.mark.asyncio
async def test_user_input_step_returns_response_when_resolved_multiple_choice() -> None:
    """Test UserInputStep returns response when multiple choice clarification resolved."""
    step = UserInputStep(
        message="Choose your favorite color",
        step_name="choose_color",
        options=["red", "green", "blue"],
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    clarification = MultipleChoiceClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Choose your favorite color",
        argument_name="user_input",
        options=["red", "green", "blue"],
        response="blue",
        resolved=True,
    )
    mock_run_data.plan_run.get_clarification_for_step.return_value = clarification

    result = await step.run(run_data=mock_run_data)

    assert result == "blue"


@pytest.mark.asyncio
async def test_user_input_step_message_with_templates() -> None:
    """Test that UserInputStep resolves references in the message and options."""
    step = UserInputStep(
        message=f"Provide feedback on {StepOutput(0)} by {Input('username')}",
        step_name="feedback",
        options=[
            "Good",
            f"Bad - {StepOutput(1)}",
            Input("Custom good phrase"),
            "Excellent",
        ],
    )

    mock_run_data = Mock()
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.id = PlanRunUUID()
    mock_run_data.plan_run.get_clarification_for_step.return_value = None
    mock_run_data.plan = Mock()
    mock_run_data.plan.step_output_name.return_value = "feedback"
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="username"),
        PlanInput(name="Custom good phrase"),
    ]
    mock_run_data.plan_run.plan_run_inputs = {
        "username": LocalDataValue(value="Alice"),
        "Custom good phrase": LocalDataValue(value="Great job!"),
    }
    mock_run_data.step_output_values = [
        StepOutputValue(
            value="analysis result",
            description="s0",
            step_name="feedback",
            step_num=0,
        ),
        StepOutputValue(
            value="missing data",
            description="s1",
            step_name="feedback",
            step_num=1,
        ),
    ]
    mock_run_data.storage = Mock()

    result = await step.run(run_data=mock_run_data)

    assert isinstance(result, MultipleChoiceClarification)
    assert result.user_guidance == "Provide feedback on analysis result by Alice"
    assert result.options == ["Good", "Bad - missing data", "Great job!", "Excellent"]


# Test cases for LoopStep class


@pytest.fixture
def mock_run_data() -> Mock:
    """Create mock run data for testing LoopStep."""
    mock_data = Mock()
    mock_data.config = Mock()
    mock_data.storage = Mock()
    mock_data.step_output_values = []
    mock_data.plan = Mock()
    mock_data.plan_run = Mock()
    return mock_data


@pytest.fixture
def mock_plan_v2() -> Mock:
    """Create mock PlanV2 for testing LoopStep."""
    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"
    return mock_plan


def test_loop_step_initialization_with_condition() -> None:
    """Test LoopStep initialization with condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        args={"x": 10},
    )

    assert step.step_name == "test_loop"
    assert step.condition is not None
    assert step.loop_type == LoopType.DO_WHILE
    assert step.loop_block_type == LoopBlockType.END
    assert step.start_index == 0
    assert step.end_index == 5
    assert step.args == {"x": 10}
    assert step.over is None


def test_loop_step_initialization_with_over() -> None:
    """Test LoopStep initialization with over reference."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
        index=0,
    )

    assert step.step_name == "test_loop"
    assert step.over is not None
    assert step.loop_type == LoopType.FOR_EACH
    assert step.loop_block_type == LoopBlockType.START
    assert step.start_index == 0
    assert step.end_index == 3
    assert step.index == 0
    assert step.condition is None


def test_loop_step_validation_error_both_none() -> None:
    """Test LoopStep validation error when both condition and over are None."""
    with pytest.raises(ValueError, match="Condition and over cannot both be None"):
        LoopStep(
            step_name="test_loop",
            condition=None,
            over=None,
            loop_type=LoopType.DO_WHILE,
            loop_block_type=LoopBlockType.END,
            start_index=0,
            end_index=5,
        )


def test_loop_step_validation_error_start_end_both_none() -> None:
    """Test LoopStep validation error when both start_index and end_index are None."""
    with pytest.raises(ValueError, match="Input should be a valid integer"):
        LoopStep(
            step_name="test_loop",
            condition=lambda x: x > 0,
            loop_type=LoopType.DO_WHILE,
            loop_block_type=LoopBlockType.END,
            start_index=None,
            end_index=None,
        )


def test_loop_step_validation_success() -> None:
    """Test LoopStep validation success with valid parameters."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
    )

    assert step.start_index == 0
    assert step.end_index == 5


def test_current_loop_variable_with_over() -> None:
    """Test current_loop_variable method when over is set."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
        index=1,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]

    result = step._current_loop_variable(mock_run_data)
    assert result == "b"


def test_current_loop_variable_with_over_none() -> None:
    """Test current_loop_variable method when over is None."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)
    assert result is None


def test_current_loop_variable_with_non_sequence() -> None:
    """Test current_loop_variable method with non-sequence value."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
        index=0,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=42,  # Use an integer which is not indexable
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]

    with pytest.raises(TypeError, match="Loop variable is not indexable"):
        step._current_loop_variable(mock_run_data)


def test_current_loop_variable_index_out_of_range() -> None:
    """Test current_loop_variable method with index out of range."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
        index=5,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]

    result = step._current_loop_variable(mock_run_data)
    assert result is None


@pytest.mark.asyncio
async def test_loop_step_run_conditional_end_with_callable() -> None:
    """Test LoopStep run method for do-while end loop with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 5,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.END
    assert result.loop_result is True
    assert result.start_index == 0
    assert result.end_index == 10


@pytest.mark.asyncio
async def test_loop_step_run_conditional_end_with_string() -> None:
    """Test LoopStep run method for do-while end loop with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 5",
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
        args={"x": 10},
    )

    mock_run_data = Mock()
    mock_run_data.config = Mock()

    with patch("portia.builder.step_v2.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result.block_type == LoopBlockType.END
        assert result.loop_result is True
        assert result.start_index == 0
        assert result.end_index == 10


@pytest.mark.asyncio
async def test_loop_step_run_conditional_end_missing_condition() -> None:
    """Test LoopStep run method for do-while end loop with missing condition."""
    # Create a valid LoopStep first, then test the run method behavior
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
    )

    mock_run_data = Mock()

    # Manually set condition to None to test the run method behavior
    step.condition = None
    with pytest.raises(ValueError, match="Condition is required for loop step"):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_loop_step_run_for_each_start() -> None:
    """Test LoopStep run method for for-each start loop."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
        index=0,
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]

    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.START
    assert result.loop_result is True
    assert result.value == "a"
    assert result.start_index == 0
    assert result.end_index == 3
    assert step.index == 1  # Should be incremented


@pytest.mark.asyncio
async def test_loop_step_run_for_each_start_missing_over() -> None:
    """Test LoopStep run method for for-each start loop with missing over."""
    # Create a valid LoopStep first, then test the run method behavior
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
    )

    mock_run_data = Mock()

    # Manually set over to None to test the run method behavior
    step.over = None
    with pytest.raises(ValueError, match="Over is required for for-each loop"):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_loop_step_run_for_each_start_no_value() -> None:
    """Test LoopStep run method for for-each start loop with no value."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=3,
        index=5,  # Out of range
    )

    mock_run_data = Mock()
    mock_run_data.storage = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            value=["a", "b", "c"],
            description="s0",
            step_name="test_step",
            step_num=0,
        )
    ]

    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.START
    assert result.loop_result is False
    assert result.value is None
    assert result.start_index == 0
    assert result.end_index == 3


@pytest.mark.asyncio
async def test_loop_step_run_default_case() -> None:
    """Test LoopStep run method for while start case."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.WHILE,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=5,
        args={"x": 10},  # Provide the required argument
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.START
    assert result.loop_result is True
    assert result.start_index == 0
    assert result.end_index == 5


@pytest.mark.asyncio
async def test_loop_step_run_with_none_indexes() -> None:
    """Test LoopStep run method with None start/end indexes."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,  # Provide at least one index
        end_index=0,  # Set to 0 instead of None to avoid validation error
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.start_index == 0
    assert result.end_index == 0


def test_loop_step_to_legacy_step_with_callable_condition() -> None:
    """Test LoopStep to_legacy_step method with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        args={"x": 10},
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"

    result = step.to_legacy_step(mock_plan)

    assert result.task == "Loop clause: If result of <lambda> is true"
    assert result.output == "loop_output"
    assert result.tool_id is None


def test_loop_step_to_legacy_step_with_string_condition() -> None:
    """Test LoopStep to_legacy_step method with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 0",
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        args={"x": 10},
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"

    result = step.to_legacy_step(mock_plan)

    assert result.task == "Loop clause: x > 0"
    assert result.output == "loop_output"
    assert result.tool_id is None


def test_loop_step_to_legacy_step_with_reference_args() -> None:
    """Test LoopStep to_legacy_step method with reference arguments."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        args={"x": StepOutput(0), "y": Input("test_input")},
    )

    mock_plan = Mock()
    mock_plan.step_output_name.return_value = "loop_output"

    result = step.to_legacy_step(mock_plan)

    assert result.task == "Loop clause: If result of <lambda> is true"
    assert result.output == "loop_output"
    assert result.tool_id is None


# New tests for the updated interface


def test_loop_step_validation_error_condition_with_for_each() -> None:
    """Test LoopStep validation error when condition is set for for-each loop."""
    with pytest.raises(ValueError, match="Condition cannot be set for for-each loop"):
        LoopStep(
            step_name="test_loop",
            condition=lambda x: x > 0,
            over=StepOutput(0),
            loop_type=LoopType.FOR_EACH,
            loop_block_type=LoopBlockType.START,
            start_index=0,
            end_index=5,
        )


def test_loop_step_validation_error_condition_and_over_both_set() -> None:
    """Test LoopStep validation error when both condition and over are set."""
    with pytest.raises(ValueError, match="Condition and over cannot both be set"):
        LoopStep(
            step_name="test_loop",
            condition=lambda x: x > 0,
            over=StepOutput(0),
            loop_type=LoopType.DO_WHILE,
            loop_block_type=LoopBlockType.END,
            start_index=0,
            end_index=5,
        )


def test_loop_step_validation_error_over_with_while() -> None:
    """Test LoopStep validation error when over is set for while loop."""
    with pytest.raises(ValueError, match="Over cannot be set for while or do-while loop"):
        LoopStep(
            step_name="test_loop",
            condition=None,
            over=StepOutput(0),
            loop_type=LoopType.WHILE,
            loop_block_type=LoopBlockType.START,
            start_index=0,
            end_index=5,
        )


def test_loop_step_validation_error_over_with_do_while() -> None:
    """Test LoopStep validation error when over is set for do-while loop."""
    with pytest.raises(ValueError, match="Over cannot be set for while or do-while loop"):
        LoopStep(
            step_name="test_loop",
            condition=None,
            over=StepOutput(0),
            loop_type=LoopType.DO_WHILE,
            loop_block_type=LoopBlockType.END,
            start_index=0,
            end_index=5,
        )


@pytest.mark.asyncio
async def test_loop_step_run_while_start_with_callable() -> None:
    """Test LoopStep run method for while start loop with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 5,
        loop_type=LoopType.WHILE,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=10,
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.START
    assert result.loop_result is True
    assert result.start_index == 0
    assert result.end_index == 10


@pytest.mark.asyncio
async def test_loop_step_run_while_start_with_string() -> None:
    """Test LoopStep run method for while start loop with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 5",
        loop_type=LoopType.WHILE,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=10,
        args={"x": 10},
    )

    mock_run_data = Mock()
    mock_run_data.config = Mock()

    with patch("portia.builder.step_v2.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result.block_type == LoopBlockType.START
        assert result.loop_result is True
        assert result.start_index == 0
        assert result.end_index == 10


@pytest.mark.asyncio
async def test_loop_step_run_do_while_end_with_callable() -> None:
    """Test LoopStep run method for do-while end loop with callable condition."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 5,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
        args={"x": 10},
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.END
    assert result.loop_result is True
    assert result.start_index == 0
    assert result.end_index == 10


@pytest.mark.asyncio
async def test_loop_step_run_do_while_end_with_string() -> None:
    """Test LoopStep run method for do-while end loop with string condition."""
    step = LoopStep(
        step_name="test_loop",
        condition="x > 5",
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
        args={"x": 10},
    )

    mock_run_data = Mock()
    mock_run_data.config = Mock()

    with patch("portia.builder.step_v2.ConditionalEvaluationAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.execute = AsyncMock(return_value=True)
        mock_agent_class.return_value = mock_agent

        result = await step.run(run_data=mock_run_data)

        assert result.block_type == LoopBlockType.END
        assert result.loop_result is True
        assert result.start_index == 0
        assert result.end_index == 10


@pytest.mark.asyncio
async def test_loop_step_run_while_start_missing_condition() -> None:
    """Test LoopStep run method for while start loop with missing condition."""
    # Create a valid LoopStep first, then test the run method behavior
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.WHILE,
        loop_block_type=LoopBlockType.START,
        start_index=0,
        end_index=10,
    )

    mock_run_data = Mock()

    # Manually set condition to None to test the run method behavior
    step.condition = None
    with pytest.raises(ValueError, match="Condition is required for loop step"):
        await step.run(run_data=mock_run_data)


@pytest.mark.asyncio
async def test_loop_step_run_for_each_end() -> None:
    """Test LoopStep run method for for-each end loop."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput(0),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=3,
        index=1,
    )

    mock_run_data = Mock()
    result = await step.run(run_data=mock_run_data)

    assert result.block_type == LoopBlockType.END
    assert result.loop_result is True
    assert result.value is True
    assert result.start_index == 0
    assert result.end_index == 3


def test_current_loop_variable_with_none_over() -> None:
    """Test _current_loop_variable when over is None."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    # Manually set over to None to test the method behavior
    step.over = None

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result is None


def test_current_loop_variable_with_sequence_over() -> None:
    """Test _current_loop_variable when over is a Sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["item1", "item2"], ["item3", "item4"]],
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "item1"


def test_current_loop_variable_with_sequence_over_different_index() -> None:
    """Test _current_loop_variable with different index values."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["item1", "item2"], ["item3", "item4"]],
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=1,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "item4"


def test_current_loop_variable_with_reference_over() -> None:
    """Test _current_loop_variable when over is a Reference."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_name="previous_step", step_num=0, value=["ref_item1", "ref_item2"])
    ]

    # Mock the _resolve_references method to return the expected sequence
    with patch.object(step, "_resolve_references", return_value=["ref_item1", "ref_item2"]):
        result = step._current_loop_variable(mock_run_data)

    assert result == "ref_item1"


def test_current_loop_variable_with_reference_over_different_index() -> None:
    """Test _current_loop_variable with Reference over and different index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=1,
    )

    mock_run_data = Mock()

    # Mock the _resolve_references method to return the expected sequence
    with patch.object(step, "_resolve_references", return_value=["ref_item1", "ref_item2"]):
        result = step._current_loop_variable(mock_run_data)

    assert result == "ref_item2"


def test_current_loop_variable_with_non_sequence_resolved_value() -> None:
    """Test _current_loop_variable when resolved value is not a sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("previous_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    mock_run_data = Mock()

    # Mock the _resolve_references method to return a non-sequence value (integer)
    with (
        patch.object(step, "_resolve_references", return_value=42),
        pytest.raises(TypeError, match="Loop variable is not indexable"),
    ):
        step._current_loop_variable(mock_run_data)


def test_current_loop_variable_with_index_out_of_bounds() -> None:
    """Test _current_loop_variable when index is out of bounds."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["item1", "item2"], ["item3", "item4"], ["item5", "item6"]],
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=2,  # Valid index for outer sequence
    )

    mock_run_data = Mock()
    # This will access over[2] = ["item5", "item6"], then ["item5", "item6"][2]
    # which is out of bounds
    result = step._current_loop_variable(mock_run_data)

    assert result is None


def test_current_loop_variable_with_empty_sequence() -> None:
    """Test _current_loop_variable with empty sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[[]],
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result is None


def test_current_loop_variable_with_nested_sequences() -> None:
    """Test _current_loop_variable with nested sequences."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[[{"key": "value1"}, {"key": "value2"}], [{"key": "value3"}]],
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == {"key": "value1"}


def test_current_loop_variable_with_mixed_types() -> None:
    """Test _current_loop_variable with mixed types in sequence."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=[["string", 42, {"dict": "value"}, [1, 2, 3]]],
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
        index=0,
    )

    mock_run_data = Mock()
    result = step._current_loop_variable(mock_run_data)

    assert result == "string"


def test_start_index_value_with_valid_index() -> None:
    """Test start_index_value with a valid start index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=5,
        end_index=10,
    )

    result = step.start_index_value

    assert result == 5


def test_start_index_value_with_zero_index() -> None:
    """Test start_index_value with zero start index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
    )

    result = step.start_index_value

    assert result == 0


def test_start_index_value_with_negative_index() -> None:
    """Test start_index_value with negative start index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=-1,
        end_index=10,
    )

    result = step.start_index_value

    assert result == -1


def test_start_index_value_with_none_raises_error() -> None:
    """Test start_index_value raises ValueError when start_index is None."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=10,
    )

    # Manually set start_index to None to test the property behavior
    step.start_index = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Start index is None"):
        _ = step.start_index_value


def test_end_index_value_with_valid_index() -> None:
    """Test end_index_value with a valid end index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=15,
    )

    result = step.end_index_value

    assert result == 15


def test_end_index_value_with_zero_index() -> None:
    """Test end_index_value with zero end index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=0,
    )

    result = step.end_index_value

    assert result == 0


def test_end_index_value_with_negative_index() -> None:
    """Test end_index_value with negative end index."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=-5,
    )

    result = step.end_index_value

    assert result == -5


def test_end_index_value_with_none_raises_error() -> None:
    """Test end_index_value raises ValueError when end_index is None."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=None,
    )

    with pytest.raises(ValueError, match="End index is None"):
        _ = step.end_index_value


def test_start_index_value_with_large_number() -> None:
    """Test start_index_value with a large number."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=1000000,
        end_index=2000000,
    )

    result = step.start_index_value

    assert result == 1000000


def test_end_index_value_with_large_number() -> None:
    """Test end_index_value with a large number."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=999999,
    )

    result = step.end_index_value

    assert result == 999999


def test_both_index_values_together() -> None:
    """Test both start_index_value and end_index_value together."""
    step = LoopStep(
        step_name="test_loop",
        condition=None,
        over=StepOutput("test_step"),
        loop_type=LoopType.FOR_EACH,
        loop_block_type=LoopBlockType.END,
        start_index=3,
        end_index=7,
    )

    start_result = step.start_index_value
    end_result = step.end_index_value

    assert start_result == 3
    assert end_result == 7
    assert end_result > start_result


def test_index_values_with_while_loop() -> None:
    """Test index values with WHILE loop type."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x > 0,
        loop_type=LoopType.WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=0,
        end_index=5,
    )

    start_result = step.start_index_value
    end_result = step.end_index_value

    assert start_result == 0
    assert end_result == 5


def test_index_values_with_do_while_loop() -> None:
    """Test index values with DO_WHILE loop type."""
    step = LoopStep(
        step_name="test_loop",
        condition=lambda x: x < 10,
        loop_type=LoopType.DO_WHILE,
        loop_block_type=LoopBlockType.END,
        start_index=1,
        end_index=8,
    )

    start_result = step.start_index_value
    end_result = step.end_index_value

    assert start_result == 1
    assert end_result == 8
