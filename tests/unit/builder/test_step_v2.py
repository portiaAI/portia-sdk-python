"""Test the step_v2 module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.reference import Input, ReferenceValue, StepOutput
from portia.builder.step_v2 import (
    InvokeToolStep,
    LLMStep,
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


class TestStepV2Base:
    """Test cases for the base StepV2 class."""

    def test_step_v2_initialization(self) -> None:
        """Test StepV2 initialization."""
        step = ConcreteStepV2("my_step")
        assert step.step_name == "my_step"

    def test_resolve_input_reference_with_non_reference(self) -> None:
        """Test _resolve_input_reference with non-reference input."""
        step = ConcreteStepV2()
        mock_run_data = Mock()

        result = step._resolve_input_reference("plain_string", mock_run_data)
        assert result == "plain_string"

        result = step._resolve_input_reference(42, mock_run_data)
        assert result == 42

    def test_resolve_input_reference_with_reference(self) -> None:
        """Test _resolve_input_reference with Reference input."""
        step = ConcreteStepV2()
        mock_run_data = Mock()
        reference = StepOutput(0)

        with patch.object(
            reference, "get_value", return_value="reference_result"
        ) as mock_get_value:
            result = step._resolve_input_reference(reference, mock_run_data)

            assert result == "reference_result"
            mock_get_value.assert_called_once_with(mock_run_data)

    def test_resolve_input_reference_with_string_template_step_output(self) -> None:
        """Test _resolve_input_reference with string containing StepOutput template."""
        step = ConcreteStepV2()
        mock_run_data = Mock()
        mock_run_data.storage = Mock()
        mock_run_data.step_output_values = [
            StepOutputValue(
                value=LocalDataValue(value="step result"),
                description="Step 0",
                step_name="test_step",
                step_num=0,
            )
        ]

        template = f"The result was {StepOutput(0)}"
        result = step._resolve_input_reference(template, mock_run_data)

        assert result == "The result was step result"

    def test_resolve_input_reference_with_string_template_input(self) -> None:
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
        result = step._resolve_input_reference(template, mock_run_data)

        assert result == "Hello Alice"

    def test_get_value_for_input_with_reference_value(self) -> None:
        """Test _get_value_for_input with ReferenceValue."""
        step = ConcreteStepV2()
        mock_run_data = Mock()
        mock_run_data.storage = Mock()

        reference_input = StepOutput(0)

        mock_data_value = LocalDataValue(value="extracted_value")
        mock_reference_value = ReferenceValue(value=mock_data_value, description="Step 0")

        with patch.object(reference_input, "get_value") as mock_get_value:
            mock_get_value.return_value = mock_reference_value

            result = step._get_value_for_input(reference_input, mock_run_data)

            assert result == "extracted_value"
            mock_get_value.assert_called_once_with(mock_run_data)

    def test_get_value_for_input_with_regular_value(self) -> None:
        """Test _get_value_for_input with regular value."""
        step = ConcreteStepV2()
        mock_run_data = Mock()

        result = step._get_value_for_input("regular_value", mock_run_data)
        assert result == "regular_value"

    def test_resolve_input_names_for_printing_with_reference(self) -> None:
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

    def test_resolve_input_names_for_printing_with_reference_already_prefixed(self) -> None:
        """Test _resolve_input_names_for_printing with Reference that already has $ prefix."""
        step = ConcreteStepV2()
        mock_plan = Mock()
        mock_reference = StepOutput(0)

        with patch.object(mock_reference, "get_legacy_name", return_value="$step_0_output"):
            result = step._resolve_input_names_for_printing(mock_reference, mock_plan)

            assert result == "$step_0_output"

    def test_resolve_input_names_for_printing_with_list(self) -> None:
        """Test _resolve_input_names_for_printing with list."""
        step = ConcreteStepV2()
        mock_plan = Mock()
        reference = Input("test_input")

        with patch.object(reference, "get_legacy_name", return_value="input_name"):
            input_list = ["regular_value", reference, 42]
            result = step._resolve_input_names_for_printing(input_list, mock_plan)

            assert result == ["regular_value", "$input_name", 42]

    def test_resolve_input_names_for_printing_with_regular_value(self) -> None:
        """Test _resolve_input_names_for_printing with regular value."""
        step = ConcreteStepV2()
        mock_plan = Mock()

        result = step._resolve_input_names_for_printing("regular_value", mock_plan)
        assert result == "regular_value"

    def test_inputs_to_legacy_plan_variables(self) -> None:
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


class TestLLMStep:
    """Test cases for the LLMStep class."""

    def test_llm_step_initialization(self) -> None:
        """Test LLMStep initialization."""
        step = LLMStep(task="Test task", step_name="llm_step")

        assert step.task == "Test task"
        assert step.step_name == "llm_step"
        assert step.inputs == []
        assert step.output_schema is None

    def test_llm_step_initialization_with_all_parameters(self) -> None:
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

    def test_llm_step_str(self) -> None:
        """Test LLMStep str method."""
        step = LLMStep(task="Test task", step_name="test")
        assert str(step) == "LLMStep(task='Test task')"

    def test_llm_step_str_with_output_schema(self) -> None:
        """Test LLMStep str method with output schema."""
        step = LLMStep(task="Test task", step_name="test", output_schema=MockOutputSchema)
        assert str(step) == "LLMStep(task='Test task' -> MockOutputSchema)"

    @pytest.mark.asyncio
    async def test_llm_step_run_no_inputs(self) -> None:
        """Test LLMStep run with no inputs."""
        step = LLMStep(task="Analyze data", step_name="analysis")
        mock_run_data = Mock()
        mock_run_data.storage = Mock()

        with (
            patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
            patch("portia.builder.step_v2.ToolRunContext"),
        ):
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
    async def test_llm_step_run_one_regular_input(self) -> None:
        """Test LLMStep run with one regular value input."""
        step = LLMStep(task="Process text", step_name="process", inputs=["Hello world"])
        mock_run_data = Mock()
        mock_run_data.storage = Mock()

        with (
            patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
            patch("portia.builder.step_v2.ToolRunContext"),
        ):
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
    async def test_llm_step_run_one_reference_input(self) -> None:
        """Test LLMStep run with one reference input."""
        reference_input = StepOutput(0)
        step = LLMStep(task="Summarize result", step_name="summarize", inputs=[reference_input])
        mock_run_data = Mock()
        mock_run_data.storage = Mock()

        mock_data_value = LocalDataValue(value="Previous step result")
        mock_reference_value = ReferenceValue(value=mock_data_value, description="Step 0")

        with (
            patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
            patch("portia.builder.step_v2.ToolRunContext"),
            patch.object(reference_input, "get_value") as mock_get_value,
        ):
            mock_get_value.return_value = mock_reference_value
            mock_wrapper_instance = Mock()
            mock_wrapper_instance.arun = AsyncMock(return_value="Summary: Previous step result")
            mock_tool_wrapper_class.return_value = mock_wrapper_instance

            result = await step.run(run_data=mock_run_data)

            assert result == "Summary: Previous step result"
            mock_wrapper_instance.arun.assert_called_once()
            call_args = mock_wrapper_instance.arun.call_args
            assert call_args[1]["task"] == "Summarize result"
            expected_task_data = "Previous step Step 0 had output: Previous step result"
            assert call_args[1]["task_data"] == [expected_task_data]

    @pytest.mark.asyncio
    async def test_llm_step_run_mixed_inputs(self) -> None:
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

        mock_data_value1 = LocalDataValue(value="John")
        mock_ref1_value = ReferenceValue(value=mock_data_value1, description="User input")

        mock_data_value2 = LocalDataValue(value="Analysis complete")
        mock_ref2_value = ReferenceValue(value=mock_data_value2, description="Step 1")

        with (
            patch("portia.builder.step_v2.ToolCallWrapper") as mock_tool_wrapper_class,
            patch("portia.builder.step_v2.ToolRunContext"),
            patch.object(ref1, "get_value") as mock_get_value1,
            patch.object(ref2, "get_value") as mock_get_value2,
        ):
            mock_get_value1.return_value = mock_ref1_value
            mock_get_value2.return_value = mock_ref2_value
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
                "Previous step User input had output: John",
                "Additional data",
                "Previous step Step 1 had output: Analysis complete",
            ]
            assert call_args[1]["task_data"] == expected_task_data

    @pytest.mark.asyncio
    async def test_llm_step_run_with_prompt(self) -> None:
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
            patch("portia.builder.step_v2.ToolRunContext"),
        ):
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
    async def test_llm_step_run_without_prompt(self) -> None:
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
            patch("portia.builder.step_v2.ToolRunContext"),
        ):
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

    async def test_llm_step_run_with_string_template_input(self) -> None:
        """Test LLMStep run with an input string containing reference templates."""
        step = LLMStep(
            task="Summarize",
            step_name="summary",
            inputs=[f"Use {StepOutput(0)} and {Input('username')}"],
        )
        mock_run_data = Mock()
        mock_run_data.step_output_values = [
            StepOutputValue(
                value=LocalDataValue(value="step0"),
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
            patch("portia.builder.step_v2.ToolRunContext"),
        ):
            mock_wrapper_instance = Mock()
            mock_wrapper_instance.arun = AsyncMock(return_value="done")
            mock_tool_wrapper_class.return_value = mock_wrapper_instance

            result = await step.run(mock_run_data)

            mock_wrapper_instance.arun.assert_called_once()
            assert result == "done"
            call_args = mock_wrapper_instance.arun.call_args
            assert call_args[1]["task_data"] == ["Use step0 and Alice"]

    def test_llm_step_to_legacy_step(self) -> None:
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


class TestInvokeToolStep:
    """Test cases for the InvokeToolStep class."""

    def test_invoke_tool_step_initialization_with_string_tool(self) -> None:
        """Test InvokeToolStep initialization with string tool."""
        step = InvokeToolStep(tool="search_tool", step_name="search")

        assert step.tool == "search_tool"
        assert step.step_name == "search"
        assert step.args == {}
        assert step.output_schema is None

    def test_invoke_tool_step_initialization_with_tool_instance(self) -> None:
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

    def test_invoke_tool_step_str_with_string_tool(self) -> None:
        """Test InvokeToolStep str method with string tool."""
        step = InvokeToolStep(
            tool="search_tool",
            step_name="search",
            args={"query": "test"},
        )

        assert str(step) == "InvokeToolStep(tool='search_tool', args={'query': 'test'})"

    def test_invoke_tool_step_str_with_tool_instance(self) -> None:
        """Test InvokeToolStep str method with Tool instance."""
        mock_tool = MockTool()
        step = InvokeToolStep(
            tool=mock_tool,
            step_name="search",
            args={"query": "test"},
            output_schema=MockOutputSchema,
        )

        assert (
            str(step)
            == "InvokeToolStep(tool='mock_tool', args={'query': 'test'} -> MockOutputSchema)"
        )

    def test_tool_name_with_string_tool(self) -> None:
        """Test _tool_name method with string tool."""
        step = InvokeToolStep(tool="search_tool", step_name="search")
        assert step._tool_name() == "search_tool"

    def test_tool_name_with_tool_instance(self) -> None:
        """Test _tool_name method with Tool instance."""
        mock_tool = MockTool()
        step = InvokeToolStep(tool=mock_tool, step_name="search")
        assert step._tool_name() == "mock_tool"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_regular_value_input(self) -> None:
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
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_ctx_class.return_value = Mock()

            result = await step.run(run_data=mock_run_data)

            assert result == "tool result"
            mock_get_tool.assert_called_once_with(
                "mock_tool",
                mock_run_data.tool_registry,
                mock_run_data.storage,
                mock_run_data.plan_run,
            )
            mock_tool._arun.assert_called_once()
            call_args = mock_tool._arun.call_args
            assert call_args[1]["query"] == "search term"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_regular_value_input_and_output_schema(self) -> None:
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
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_model.return_value = mock_model
            mock_ctx_class.return_value = Mock()

            result = await step.run(run_data=mock_run_data)

            assert isinstance(result, MockOutputSchema)
            assert result.result == "structured result"
            mock_get_tool.assert_called_once_with(
                "mock_tool",
                mock_run_data.tool_registry,
                mock_run_data.storage,
                mock_run_data.plan_run,
            )
            mock_tool._arun.assert_called_once()
            mock_model.aget_structured_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_tool_output_schema(self) -> None:
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
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_model.return_value = mock_model
            mock_ctx_class.return_value = Mock()

            result = await step.run(run_data=mock_run_data)

            assert isinstance(result, MockOutputSchema)
            assert result.result == "structured result"
            mock_get_tool.assert_called_once_with(
                "mock_tool",
                mock_run_data.tool_registry,
                mock_run_data.storage,
                mock_run_data.plan_run,
            )
            mock_tool._arun.assert_called_once()
            mock_model.aget_structured_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_reference_input(self) -> None:
        """Test InvokeToolStep run with 1 reference input."""
        reference_input = StepOutput(0)
        step = InvokeToolStep(
            tool="mock_tool", step_name="run_tool", args={"query": reference_input}
        )
        mock_run_data = Mock()
        mock_run_data.storage = Mock()
        mock_tool = Mock()
        mock_tool.structured_output_schema = None
        mock_output = Mock()
        mock_output.get_value.return_value = "tool result with reference"
        mock_tool._arun = AsyncMock(return_value=mock_output)

        # Create proper ReferenceValue that the real methods can work with
        mock_data_value = LocalDataValue(value="previous step output")
        mock_reference_value = ReferenceValue(value=mock_data_value, description="Step 0")

        with (
            patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
            patch.object(reference_input, "get_value") as mock_get_value,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_value.return_value = mock_reference_value
            mock_ctx_class.return_value = Mock()

            result = await step.run(run_data=mock_run_data)

            assert result == "tool result with reference"
            mock_get_tool.assert_called_once_with(
                "mock_tool",
                mock_run_data.tool_registry,
                mock_run_data.storage,
                mock_run_data.plan_run,
            )
            mock_tool._arun.assert_called_once()
            call_args = mock_tool._arun.call_args
            assert call_args[1]["query"] == "previous step output"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_mixed_inputs(self) -> None:
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

        mock_data_value1 = LocalDataValue(value="user question")
        mock_ref1_value = ReferenceValue(value=mock_data_value1, description="User input")

        mock_data_value2 = LocalDataValue(value="step 1 output")
        mock_ref2_value = ReferenceValue(value=mock_data_value2, description="Step 1")

        with (
            patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
            patch.object(ref1, "get_value") as mock_get_value1,
            patch.object(ref2, "get_value") as mock_get_value2,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_value1.return_value = mock_ref1_value
            mock_get_value2.return_value = mock_ref2_value
            mock_ctx_class.return_value = Mock()

            result = await step.run(run_data=mock_run_data)

            assert result == "mixed inputs result"
            mock_get_tool.assert_called_once_with(
                "mock_tool",
                mock_run_data.tool_registry,
                mock_run_data.storage,
                mock_run_data.plan_run,
            )
            mock_tool._arun.assert_called_once()
            call_args = mock_tool._arun.call_args[1]
            assert call_args["context"] == "static context"
            assert call_args["user_input"] == "user question"
            assert call_args["limit"] == 10
            assert call_args["previous_result"] == "step 1 output"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_no_args_with_clarification(self) -> None:
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
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_ctx_class.return_value = Mock()

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
            mock_tool._arun.assert_called_once_with(mock_ctx_class.return_value)

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_tool_instance(self) -> None:
        """Test InvokeToolStep run with Tool instance instead of string tool."""
        mock_tool = MockTool()
        step = InvokeToolStep(tool=mock_tool, step_name="run_tool", args={"input": "test input"})
        mock_run_data = Mock()
        mock_run_data.plan_run.id = PlanRunUUID()
        mock_run_data.plan_run.current_step_index = 0
        mock_run_data.storage = AsyncMock()

        with (
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
            patch.object(mock_tool, "_arun") as mock_arun,
        ):
            mock_ctx = Mock()
            mock_ctx.end_user.external_id = "test_user_id"
            mock_ctx_class.return_value = mock_ctx
            mock_output = Mock()
            mock_output.get_value.return_value = "mock result"
            mock_arun.return_value = mock_output

            result = await step.run(run_data=mock_run_data)

            assert result == "mock result"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_nonexistent_tool_id(self) -> None:
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
    async def test_invoke_tool_step_with_function_tool(self) -> None:
        """Test InvokeToolStep run with nonexistent tool_id raises ToolNotFoundError."""
        tool_class = tool(example_function)
        step = InvokeToolStep(
            tool=tool_class(), step_name="run_tool", args={"x": 42, "y": "Result"}
        )
        mock_run_data = Mock()
        mock_run_data.plan_run.id = PlanRunUUID()
        mock_run_data.plan_run.current_step_index = 0
        mock_run_data.storage = AsyncMock()

        with patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class:
            mock_ctx = Mock()
            mock_ctx.end_user.external_id = "test_user_id"
            mock_ctx_class.return_value = mock_ctx
            result = await step.run(run_data=mock_run_data)

            assert result == "Result: 42"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_async_function_tool(self) -> None:
        """Test InvokeToolStep run with nonexistent tool_id raises ToolNotFoundError."""

        async def async_example_function(x: int, y: str) -> str:
            await asyncio.sleep(0.001)
            return f"{y}: {x}"

        tool_class = tool(async_example_function)
        step = InvokeToolStep(
            tool=tool_class(), step_name="run_tool", args={"x": 42, "y": "Result"}
        )
        mock_run_data = Mock()
        # Configure mock to return proper values for ToolCallRecord
        mock_run_data.plan_run.id = PlanRunUUID()
        mock_run_data.plan_run.current_step_index = 0
        mock_run_data.storage = AsyncMock()

        with patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class:
            mock_ctx = Mock()
            mock_ctx.end_user.external_id = "test_user_id"
            mock_ctx_class.return_value = mock_ctx
            result = await step.run(run_data=mock_run_data)

            assert result == "Result: 42"

    @pytest.mark.asyncio
    async def test_invoke_tool_step_with_string_arg_templates(self) -> None:
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
                value=LocalDataValue(value="result"),
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
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "final"
            call_args = mock_tool._arun.call_args
            assert call_args[1]["query"] == "Search result for Alice"

    def test_invoke_tool_step_to_legacy_step(self) -> None:
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


class TestSingleToolAgent:
    """Test cases for the SingleToolAgent class."""

    def test_single_tool_agent_initialization(self) -> None:
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

    def test_single_tool_agent_str(self) -> None:
        """Test SingleToolAgent str method."""
        step = SingleToolAgentStep(
            task="Search for info",
            tool="search_tool",
            step_name="search",
        )

        assert str(step) == "SingleToolAgentStep(tool='search_tool', query='Search for info')"

    def test_single_tool_agent_str_with_output_schema(self) -> None:
        """Test SingleToolAgent str method with output schema."""
        step = SingleToolAgentStep(
            task="Search for info",
            tool="search_tool",
            step_name="search",
            output_schema=MockOutputSchema,
        )

        expected_str = (
            "SingleToolAgentStep(tool='search_tool', query='Search for info' -> MockOutputSchema)"
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
        self, execution_agent_type: ExecutionAgentType, expected_one_shot: bool
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
        mock_output = Mock()
        mock_output.get_value.return_value = "Agent execution result"

        with (
            patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
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
            mock_ctx_class.return_value = Mock()

            result = await step.run(run_data=mock_run_data)

            assert result == "Agent execution result"

            if expected_one_shot:
                mock_oneshot_execute.assert_called_once()
                mock_default_execute.assert_not_called()
            else:
                mock_default_execute.assert_called_once()
                mock_oneshot_execute.assert_not_called()

    def test_single_tool_agent_to_legacy_step(self) -> None:
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
    async def test_single_tool_agent_with_string_template_task_and_inputs(self) -> None:
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
                value=LocalDataValue(value="machine learning"),
                description="step0",
            ),
            StepOutputValue(
                step_num=1,
                step_name="step1",
                value=LocalDataValue(value="AI research"),
                description="step1",
            ),
        ]

        # Create mock agent and output object
        mock_output_obj = Mock()
        mock_output_obj.get_value.return_value = "Search completed successfully"
        mock_tool = Mock()

        # Mock the plan for to_legacy_step conversion
        mock_plan = Mock()
        mock_plan.step_output_name.return_value = "$templated_search_output"
        mock_run_data.plan = mock_plan

        with (
            patch("portia.builder.step_v2.ToolCallWrapper.from_tool_id") as mock_get_tool,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
            patch.object(
                OneShotAgent, "execute_async", new_callable=AsyncMock, return_value=mock_output_obj
            ),
        ):
            mock_get_tool.return_value = mock_tool
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "Search completed successfully"

            # The task should contain the original template string (not resolved yet)
            # Template resolution happens within the execution agent
            expected_task = (
                f"Search for information about {StepOutput(0)} requested by {Input('username')}"
            )
            assert step.task == expected_task


class TestUserVerifyStep:
    """Test cases for the UserVerifyStep class."""

    def test_user_verify_step_str(self) -> None:
        """Test UserVerifyStep str method."""
        step = UserVerifyStep(message="Please confirm this action", step_name="verify")
        assert str(step) == "UserVerifyStep(message='Please confirm this action')"

    def test_user_verify_step_to_legacy_step(self) -> None:
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
    async def test_user_verify_step_requests_clarification(self) -> None:
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
            StepOutputValue(
                step_num=0, step_name="step_0", value=LocalDataValue(value="result"), description=""
            )
        ]

        result = await step.run(run_data=mock_run_data)

        assert isinstance(result, UserVerificationClarification)
        assert result.user_guidance == "Proceed with result for Alice?"

    @pytest.mark.asyncio
    async def test_user_verify_step_user_confirms(self) -> None:
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
    async def test_user_verify_step_user_rejects(self) -> None:
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
    async def test_user_verify_step_with_string_template_message(self) -> None:
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
                value=LocalDataValue(value="test_file.txt"),
                description="step0",
                step_name="verify_action",
                step_num=0,
            )
        ]

        result = await step.run(mock_run_data)

        assert isinstance(result, UserVerificationClarification)
        assert result.user_guidance == "Confirm action on test_file.txt for user Bob?"


class TestUserInputStep:
    """Test cases for the UserInputStep class."""

    def test_user_input_step_str_text_input(self) -> None:
        """Test UserInputStep str method for text input."""
        step = UserInputStep(
            message="Please provide input",
            step_name="input",
        )

        assert str(step) == "UserInputStep(type='text input', message='Please provide input')"

    def test_user_input_step_str_multiple_choice(self) -> None:
        """Test UserInputStep str method for multiple choice."""
        step = UserInputStep(
            message="Choose an option",
            step_name="choice",
            options=["A", "B", "C"],
        )

        assert str(step) == "UserInputStep(type='multiple choice', message='Choose an option')"

    def test_user_input_step_to_legacy_step_text_input(self) -> None:
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

    def test_user_input_step_to_legacy_step_multiple_choice(self) -> None:
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
    async def test_user_input_step_text_input_requests_clarification(self) -> None:
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
    async def test_user_input_step_multiple_choice_requests_clarification(self) -> None:
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
    async def test_user_input_step_returns_response_when_resolved_text_input(self) -> None:
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
    async def test_user_input_step_returns_response_when_resolved_multiple_choice(
        self,
    ) -> None:
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
    async def test_user_input_step_message_with_templates(self) -> None:
        """Test that UserInputStep resolves references in the message."""
        step = UserInputStep(
            message=f"Provide feedback on {StepOutput(0)} by {Input('username')}",
            step_name="feedback",
        )

        mock_run_data = Mock()
        mock_run_data.plan_run = Mock()
        mock_run_data.plan_run.id = PlanRunUUID()
        mock_run_data.plan_run.get_clarification_for_step.return_value = None
        mock_run_data.plan = Mock()
        mock_run_data.plan.step_output_name.return_value = "feedback"
        mock_run_data.plan.plan_inputs = [PlanInput(name="username")]
        mock_run_data.plan_run.plan_run_inputs = {"username": LocalDataValue(value="Alice")}
        mock_run_data.step_output_values = [
            StepOutputValue(
                value=LocalDataValue(value="result"),
                description="s0",
                step_name="feedback",
                step_num=0,
            )
        ]
        mock_run_data.storage = Mock()

        result = await step.run(mock_run_data)

        assert isinstance(result, InputClarification)
        assert result.user_guidance == "Provide feedback on result by Alice"
