"""Test the step_v2 module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from portia.builder.reference import Input, ReferenceValue, StepOutput

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import FunctionCall, LLMStep, SingleToolAgent, StepV2, ToolRun
from portia.clarification import ClarificationCategory
from portia.execution_agents.output import LocalDataValue
from portia.model import Message
from portia.plan import Step as PlanStep
from portia.plan import Variable
from portia.tool import Tool


class MockOutputSchema(BaseModel):
    """Mock output schema for testing."""

    result: str
    count: int


def example_function(x: int, y: str) -> str:
    """Return formatted string for testing FunctionCall."""
    return f"{y}: {x}"


class MockTool(Tool):
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


class ConcreteStepV2(StepV2):
    """Concrete implementation of StepV2 for testing base functionality."""

    def __init__(self, step_name: str = "test_step") -> None:
        """Initialize concrete step."""
        super().__init__(step_name=step_name)

    async def run(self, run_data: Any) -> str:  # noqa: ANN401, ARG002
        """Mock run method."""
        return "test result"

    def describe(self) -> str:
        """Mock describe method."""
        return f"ConcreteStepV2(step_name='{self.step_name}')"

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
        # TODO: Fill in this test  # noqa: FIX002, TD002, TD003

    def test_resolve_input_reference_with_string_template_input(self) -> None:
        """Test _resolve_input_reference with string containing Input template."""
        # TODO: Fill in this test  # noqa: FIX002, TD002, TD003

    def test_get_value_for_input_with_reference_value(self) -> None:
        """Test _get_value_for_input with ReferenceValue."""
        step = ConcreteStepV2()
        mock_run_data = Mock()
        mock_run_data.portia.storage = Mock()

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

    def test_llm_step_describe(self) -> None:
        """Test LLMStep describe method."""
        step = LLMStep(task="Test task", step_name="test")
        assert step.describe() == "LLMStep(task='Test task')"

    def test_llm_step_describe_with_output_schema(self) -> None:
        """Test LLMStep describe method with output schema."""
        step = LLMStep(task="Test task", step_name="test", output_schema=MockOutputSchema)
        assert step.describe() == "LLMStep(task='Test task' -> MockOutputSchema)"

    @pytest.mark.asyncio
    async def test_llm_step_run_no_inputs(self) -> None:
        """Test LLMStep run with no inputs."""
        step = LLMStep(task="Analyze data", step_name="analysis")
        mock_run_data = Mock()

        with (
            patch("portia.builder.step_v2.LLMTool") as mock_llm_tool_class,
            patch("portia.builder.step_v2.ToolRunContext") as mock_tool_ctx_class,
        ):
            mock_llm_tool = Mock()
            mock_llm_tool.arun = AsyncMock(return_value="Analysis complete")
            mock_llm_tool_class.return_value = mock_llm_tool
            mock_tool_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "Analysis complete"
            mock_llm_tool_class.assert_called_once_with(structured_output_schema=None)
            mock_llm_tool.arun.assert_called_once()
            call_args = mock_llm_tool.arun.call_args
            assert call_args[1]["task"] == "Analyze data"
            assert call_args[1]["task_data"] == []

    @pytest.mark.asyncio
    async def test_llm_step_run_one_regular_input(self) -> None:
        """Test LLMStep run with one regular value input."""
        step = LLMStep(task="Process text", step_name="process", inputs=["Hello world"])
        mock_run_data = Mock()

        with (
            patch("portia.builder.step_v2.LLMTool") as mock_llm_tool_class,
            patch("portia.builder.step_v2.ToolRunContext") as mock_tool_ctx_class,
        ):
            mock_llm_tool = Mock()
            mock_llm_tool.arun = AsyncMock(return_value="Processed: Hello world")
            mock_llm_tool_class.return_value = mock_llm_tool
            mock_tool_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "Processed: Hello world"
            mock_llm_tool.arun.assert_called_once()
            call_args = mock_llm_tool.arun.call_args
            assert call_args[1]["task"] == "Process text"
            assert call_args[1]["task_data"] == ["Hello world"]

    @pytest.mark.asyncio
    async def test_llm_step_run_one_reference_input(self) -> None:
        """Test LLMStep run with one reference input."""
        reference_input = StepOutput(0)
        step = LLMStep(task="Summarize result", step_name="summarize", inputs=[reference_input])
        mock_run_data = Mock()
        mock_run_data.portia.storage = Mock()

        mock_data_value = LocalDataValue(value="Previous step result")
        mock_reference_value = ReferenceValue(value=mock_data_value, description="Step 0")

        with (
            patch("portia.builder.step_v2.LLMTool") as mock_llm_tool_class,
            patch("portia.builder.step_v2.ToolRunContext") as mock_tool_ctx_class,
            patch.object(reference_input, "get_value") as mock_get_value,
        ):
            mock_get_value.return_value = mock_reference_value
            mock_llm_tool = Mock()
            mock_llm_tool.arun = AsyncMock(return_value="Summary: Previous step result")
            mock_llm_tool_class.return_value = mock_llm_tool
            mock_tool_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "Summary: Previous step result"
            mock_llm_tool.arun.assert_called_once()
            call_args = mock_llm_tool.arun.call_args
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
        mock_run_data.portia.storage = Mock()

        mock_data_value1 = LocalDataValue(value="John")
        mock_ref1_value = ReferenceValue(value=mock_data_value1, description="User input")

        mock_data_value2 = LocalDataValue(value="Analysis complete")
        mock_ref2_value = ReferenceValue(value=mock_data_value2, description="Step 1")

        with (
            patch("portia.builder.step_v2.LLMTool") as mock_llm_tool_class,
            patch("portia.builder.step_v2.ToolRunContext") as mock_tool_ctx_class,
            patch.object(ref1, "get_value") as mock_get_value1,
            patch.object(ref2, "get_value") as mock_get_value2,
        ):
            mock_get_value1.return_value = mock_ref1_value
            mock_get_value2.return_value = mock_ref2_value
            mock_llm_tool = Mock()
            mock_llm_tool.arun = AsyncMock(return_value="Report generated successfully")
            mock_llm_tool_class.return_value = mock_llm_tool
            mock_tool_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "Report generated successfully"
            mock_llm_tool.arun.assert_called_once()
            call_args = mock_llm_tool.arun.call_args
            assert call_args[1]["task"] == "Generate report"

            expected_task_data = [
                "Context info",
                "Previous step User input had output: John",
                "Additional data",
                "Previous step Step 1 had output: Analysis complete",
            ]
            assert call_args[1]["task_data"] == expected_task_data

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


class TestToolRun:
    """Test cases for the ToolRun class."""

    def test_tool_run_initialization_with_string_tool(self) -> None:
        """Test ToolRun initialization with string tool."""
        step = ToolRun(tool="search_tool", step_name="search")

        assert step.tool == "search_tool"
        assert step.step_name == "search"
        assert step.args == {}
        assert step.output_schema is None

    def test_tool_run_initialization_with_tool_instance(self) -> None:
        """Test ToolRun initialization with Tool instance."""
        mock_tool = MockTool()
        args = {"query": "test", "limit": StepOutput(0)}

        step = ToolRun(
            tool=mock_tool,
            step_name="search",
            args=args,
            output_schema=MockOutputSchema,
        )

        assert step.tool is mock_tool
        assert step.args == args
        assert step.output_schema == MockOutputSchema

    def test_tool_run_describe_with_string_tool(self) -> None:
        """Test ToolRun describe method with string tool."""
        step = ToolRun(
            tool="search_tool",
            step_name="search",
            args={"query": "test"},
        )

        result = step.describe()
        assert result == "ToolRun(tool='search_tool', args={'query': 'test'})"

    def test_tool_run_describe_with_tool_instance(self) -> None:
        """Test ToolRun describe method with Tool instance."""
        mock_tool = MockTool()
        step = ToolRun(
            tool=mock_tool,
            step_name="search",
            args={"query": "test"},
            output_schema=MockOutputSchema,
        )

        result = step.describe()
        expected = "ToolRun(tool='mock_tool', args={'query': 'test'} -> MockOutputSchema)"
        assert result == expected

    def test_tool_name_with_string_tool(self) -> None:
        """Test _tool_name method with string tool."""
        step = ToolRun(tool="search_tool", step_name="search")
        assert step._tool_name() == "search_tool"

    def test_tool_name_with_tool_instance(self) -> None:
        """Test _tool_name method with Tool instance."""
        mock_tool = MockTool()
        step = ToolRun(tool=mock_tool, step_name="search")
        assert step._tool_name() == "mock_tool"

    @pytest.mark.asyncio
    async def test_tool_run_with_regular_value_input(self) -> None:
        """Test ToolRun run with 1 regular value input."""
        step = ToolRun(tool="mock_tool", step_name="run_tool", args={"query": "search term"})
        mock_run_data = Mock()
        mock_tool = Mock()
        mock_tool.run.return_value = "tool result"

        with (
            patch.object(mock_run_data.portia, "get_tool") as mock_get_tool,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "tool result"
            mock_get_tool.assert_called_once_with("mock_tool", mock_run_data.plan_run)
            mock_tool.run.assert_called_once()
            call_args = mock_tool.run.call_args
            assert call_args[1]["query"] == "search term"

    @pytest.mark.asyncio
    async def test_tool_run_with_regular_value_input_and_output_schema(self) -> None:
        """Test ToolRun run with 1 regular value input and output schema."""
        step = ToolRun(
            tool="mock_tool",
            step_name="run_tool",
            args={"query": "search term"},
            output_schema=MockOutputSchema,
        )
        mock_run_data = Mock()
        mock_tool = Mock()
        mock_tool.run.return_value = "raw tool result"

        # Mock the model and its aget_structured_response method
        mock_model = Mock()
        mock_model.aget_structured_response = AsyncMock(
            return_value=MockOutputSchema(result="structured result", count=5)
        )

        with (
            patch.object(mock_run_data.portia, "get_tool") as mock_get_tool,
            patch.object(mock_run_data.portia.config, "get_default_model") as mock_get_model,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_model.return_value = mock_model
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert isinstance(result, MockOutputSchema)
            assert result.result == "structured result"
            mock_get_tool.assert_called_once_with("mock_tool", mock_run_data.plan_run)
            mock_tool.run.assert_called_once()
            mock_model.aget_structured_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_run_with_reference_input(self) -> None:
        """Test ToolRun run with 1 reference input."""
        reference_input = StepOutput(0)
        step = ToolRun(tool="mock_tool", step_name="run_tool", args={"query": reference_input})
        mock_run_data = Mock()
        mock_run_data.portia.storage = Mock()
        mock_tool = Mock()
        mock_tool.run.return_value = "tool result with reference"

        # Create proper ReferenceValue that the real methods can work with
        mock_data_value = LocalDataValue(value="previous step output")
        mock_reference_value = ReferenceValue(value=mock_data_value, description="Step 0")

        with (
            patch.object(mock_run_data.portia, "get_tool") as mock_get_tool,
            patch.object(reference_input, "get_value") as mock_get_value,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_value.return_value = mock_reference_value
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "tool result with reference"
            mock_get_tool.assert_called_once_with("mock_tool", mock_run_data.plan_run)
            mock_tool.run.assert_called_once()
            call_args = mock_tool.run.call_args
            assert call_args[1]["query"] == "previous step output"

    @pytest.mark.asyncio
    async def test_tool_run_with_mixed_inputs(self) -> None:
        """Test ToolRun run with 2 regular value inputs and 2 reference inputs."""
        ref1 = Input("user_query")
        ref2 = StepOutput(1)
        step = ToolRun(
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
        mock_run_data.portia.storage = Mock()
        mock_tool = Mock()
        mock_tool.run.return_value = "mixed inputs result"

        mock_data_value1 = LocalDataValue(value="user question")
        mock_ref1_value = ReferenceValue(value=mock_data_value1, description="User input")

        mock_data_value2 = LocalDataValue(value="step 1 output")
        mock_ref2_value = ReferenceValue(value=mock_data_value2, description="Step 1")

        with (
            patch.object(mock_run_data.portia, "get_tool") as mock_get_tool,
            patch.object(ref1, "get_value") as mock_get_value1,
            patch.object(ref2, "get_value") as mock_get_value2,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_get_value1.return_value = mock_ref1_value
            mock_get_value2.return_value = mock_ref2_value
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert result == "mixed inputs result"
            mock_get_tool.assert_called_once_with("mock_tool", mock_run_data.plan_run)
            mock_tool.run.assert_called_once()
            call_args = mock_tool.run.call_args[1]
            assert call_args["context"] == "static context"
            assert call_args["user_input"] == "user question"
            assert call_args["limit"] == 10
            assert call_args["previous_result"] == "step 1 output"

    @pytest.mark.asyncio
    async def test_tool_run_no_args_with_clarification(self) -> None:
        """Test ToolRun run with no args and tool returns a clarification."""
        from portia.clarification import Clarification

        step = ToolRun(tool="mock_tool", step_name="run_tool", args={})
        mock_run_data = Mock()
        mock_tool = Mock()

        mock_clarification = Clarification(
            category=ClarificationCategory.ACTION,
            user_guidance="Need more information",
            plan_run_id=None,
        )
        mock_tool.run.return_value = mock_clarification

        with (
            patch.object(mock_run_data.portia, "get_tool") as mock_get_tool,
            patch("portia.builder.step_v2.ToolRunContext") as mock_ctx_class,
        ):
            mock_get_tool.return_value = mock_tool
            mock_ctx_class.return_value = Mock()

            result = await step.run(mock_run_data)

            assert isinstance(result, Clarification)
            assert result.user_guidance == "Need more information"
            assert result.plan_run_id == mock_run_data.plan_run.id
            mock_get_tool.assert_called_once_with("mock_tool", mock_run_data.plan_run)
            mock_tool.run.assert_called_once_with(mock_ctx_class.return_value)

    def test_tool_run_to_legacy_step(self) -> None:
        """Test ToolRun to_legacy_step method."""
        args = {"query": Input("user_query"), "limit": 10}
        step = ToolRun(
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
            assert (
                legacy_step.task == "Use tool search_tool with inputs: query=$user_query, limit=10"
            )
            assert legacy_step.tool_id == "search_tool"
            assert legacy_step.output == "$search_output"
            assert legacy_step.structured_output_schema == MockOutputSchema

            assert len(legacy_step.inputs) == 1
            assert legacy_step.inputs[0].name == "user_query"


class TestFunctionCall:
    """Test cases for the FunctionCall class."""

    def test_function_call_initialization(self) -> None:
        """Test FunctionCall initialization."""
        args = {"x": 42, "y": Input("user_input")}
        step = FunctionCall(
            function=example_function,
            step_name="calc",
            args=args,
            output_schema=MockOutputSchema,
        )

        assert step.function is example_function
        assert step.args == args
        assert step.output_schema == MockOutputSchema

    def test_function_call_describe(self) -> None:
        """Test FunctionCall describe method."""
        step = FunctionCall(
            function=example_function,
            step_name="calc",
            args={"x": 42, "y": "test"},
        )

        result = step.describe()
        expected = "FunctionCall(function='example_function', args={'x': 42, 'y': 'test'})"
        assert result == expected

    def test_function_call_describe_with_output_schema(self) -> None:
        """Test FunctionCall describe method with output schema."""
        step = FunctionCall(
            function=example_function,
            step_name="calc",
            args={"x": 42},
            output_schema=MockOutputSchema,
        )

        result = step.describe()
        expected = "FunctionCall(function='example_function', args={'x': 42} -> MockOutputSchema)"
        assert result == expected

    @pytest.mark.asyncio
    async def test_function_call_with_normal_arg(self) -> None:
        """Test FunctionCall run with normal arguments."""
        step = FunctionCall(
            function=example_function, step_name="calc", args={"x": 42, "y": "Result"}
        )
        mock_run_data = Mock()

        result = await step.run(mock_run_data)

        assert result == "Result: 42"

    @pytest.mark.asyncio
    async def test_function_call_with_reference_arg(self) -> None:
        """Test FunctionCall run with reference argument."""
        reference_input = StepOutput(0)
        step = FunctionCall(
            function=example_function, step_name="calc", args={"x": 10, "y": reference_input}
        )
        mock_run_data = Mock()
        mock_run_data.portia.storage = Mock()

        mock_data_value = LocalDataValue(value="Previous step")
        mock_reference_value = ReferenceValue(value=mock_data_value, description="Step 0")

        with patch.object(reference_input, "get_value") as mock_get_value:
            mock_get_value.return_value = mock_reference_value

            result = await step.run(mock_run_data)

            assert result == "Previous step: 10"
            mock_get_value.assert_called_once_with(mock_run_data)

    @pytest.mark.asyncio
    async def test_function_call_no_args_with_clarification(self) -> None:
        """Test FunctionCall run with no args that returns a clarification."""
        from portia.clarification import Clarification

        def clarification_function() -> Clarification:
            return Clarification(
                category=ClarificationCategory.ACTION,
                user_guidance="Function needs clarification",
                plan_run_id=None,
            )

        step = FunctionCall(function=clarification_function, step_name="clarify", args={})
        mock_run_data = Mock()

        result = await step.run(mock_run_data)

        assert isinstance(result, Clarification)
        assert result.user_guidance == "Function needs clarification"
        assert result.plan_run_id == mock_run_data.plan_run.id

    @pytest.mark.asyncio
    async def test_function_call_no_args_with_output_schema(self) -> None:
        """Test FunctionCall run with no args and output schema conversion."""

        def raw_output_function() -> str:
            return "raw function output"

        step = FunctionCall(
            function=raw_output_function,
            step_name="convert",
            args={},
            output_schema=MockOutputSchema,
        )
        mock_run_data = Mock()

        mock_model = Mock()
        mock_model.aget_structured_response = AsyncMock(
            return_value=MockOutputSchema(result="converted output", count=1)
        )

        with patch.object(mock_run_data.portia.config, "get_default_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            result = await step.run(mock_run_data)

            assert isinstance(result, MockOutputSchema)
            assert result.result == "converted output"
            assert result.count == 1
            mock_model.aget_structured_response.assert_called_once()

            # Verify the conversion message
            call_args = mock_model.aget_structured_response.call_args
            messages = call_args[0][0]
            assert len(messages) == 1
            assert isinstance(messages[0], Message)
            expected_content = "Convert this output to the desired schema: raw function output"
            assert expected_content in messages[0].content

    def test_function_call_to_legacy_step(self) -> None:
        """Test FunctionCall to_legacy_step method."""
        args = {"x": Input("number"), "y": "constant"}
        step = FunctionCall(
            function=example_function,
            step_name="calc",
            args=args,
            output_schema=MockOutputSchema,
        )

        mock_plan = Mock()
        mock_plan.step_output_name.return_value = "$calc_output"

        # Mock the get_legacy_name method on the Input reference
        with patch.object(args["x"], "get_legacy_name") as mock_input_name:
            mock_input_name.return_value = "number"

            legacy_step = step.to_legacy_step(mock_plan)

            assert isinstance(legacy_step, PlanStep)
            assert (
                legacy_step.task == "Run function example_function with args: x=$number, y=constant"
            )
            assert legacy_step.tool_id == "local_function_example_function"
            assert legacy_step.output == "$calc_output"
            assert legacy_step.structured_output_schema == MockOutputSchema

            assert len(legacy_step.inputs) == 1
            assert legacy_step.inputs[0].name == "number"


class TestSingleToolAgent:
    """Test cases for the SingleToolAgent class."""

    def test_single_tool_agent_initialization(self) -> None:
        """Test SingleToolAgent initialization."""
        inputs = [Input("query"), "context"]
        step = SingleToolAgent(
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

    def test_single_tool_agent_describe(self) -> None:
        """Test SingleToolAgent describe method."""
        step = SingleToolAgent(
            task="Search for info",
            tool="search_tool",
            step_name="search",
        )

        result = step.describe()
        expected = "SingleToolAgent(tool='search_tool', query='Search for info')"
        assert result == expected

    def test_single_tool_agent_describe_with_output_schema(self) -> None:
        """Test SingleToolAgent describe method with output schema."""
        step = SingleToolAgent(
            task="Search for info",
            tool="search_tool",
            step_name="search",
            output_schema=MockOutputSchema,
        )

        result = step.describe()
        expected = (
            "SingleToolAgent(tool='search_tool', query='Search for info' -> MockOutputSchema)"
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_single_tool_agent_run_with_mocked_agent(self) -> None:
        """Test SingleToolAgent run with get_agent_for_step and agent mocked."""
        step = SingleToolAgent(
            tool="search_tool", task="Search for information", step_name="search"
        )
        mock_run_data = Mock()

        # Create mock agent and output object
        mock_agent = Mock()
        mock_output_obj = Mock()
        mock_output_obj.get_value.return_value = "Agent execution result"
        mock_agent.execute_async = AsyncMock(return_value=mock_output_obj)

        # Need to mock the plan for to_legacy_step conversion
        mock_plan = Mock()
        mock_plan.step_output_name.return_value = "$search_output"
        mock_run_data.plan = mock_plan

        with patch.object(mock_run_data.portia, "get_agent_for_step") as mock_get_agent:
            mock_get_agent.return_value = mock_agent

            result = await step.run(mock_run_data)

            assert result == "Agent execution result"

            # Verify get_agent_for_step was called with correct arguments
            mock_get_agent.assert_called_once()
            call_args = mock_get_agent.call_args[0]
            # First argument should be the legacy step
            legacy_step = call_args[0]
            assert legacy_step.task == "Search for information"
            assert legacy_step.tool_id == "search_tool"
            # Second and third arguments should be legacy plan and plan run
            assert call_args[1] == mock_run_data.legacy_plan
            assert call_args[2] == mock_run_data.plan_run

            # Verify agent.execute_async was called
            mock_agent.execute_async.assert_called_once()

            # Verify output_obj.get_value was called
            mock_output_obj.get_value.assert_called_once()

    def test_single_tool_agent_to_legacy_step(self) -> None:
        """Test SingleToolAgent to_legacy_step method."""
        inputs = [Input("query"), StepOutput(0)]
        step = SingleToolAgent(
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
