"""Test the PlanBuilderV2 class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from portia.builder.conditional_step import ConditionalStep
from portia.builder.conditionals import ConditionalBlockClauseType
from portia.builder.invoke_tool_step import InvokeToolStep
from portia.builder.llm_step import LLMStep
from portia.builder.loop_step import LoopStep
from portia.builder.loops import LoopStepType, LoopType
from portia.builder.plan_builder_v2 import PlanBuilderError, PlanBuilderV2
from portia.builder.plan_v2 import PlanV2
from portia.builder.react_agent_step import ReActAgentStep
from portia.builder.reference import Input, StepOutput
from portia.builder.single_tool_agent_step import SingleToolAgentStep
from portia.builder.step_v2 import (
    LoopBlock,
    StepV2,
)
from portia.builder.user_input import UserInputStep
from portia.builder.user_verify import UserVerifyStep
from portia.plan import PlanInput, Step
from portia.tool import Tool
from portia.tool_decorator import tool

if TYPE_CHECKING:
    from portia.run_context import RunContext


class OutputSchema(BaseModel):
    """Output schema for testing."""

    result: str
    count: int


def example_function_for_testing(x: int, y: str) -> str:
    """Example function for function call tests."""  # noqa: D401
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


class CustomStep(StepV2):
    """Custom step for testing."""

    async def run(self, run_data: RunContext) -> Any:  # noqa: ANN401, ARG002
        """Execute the step."""
        return "mock result"

    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this step to a Step from plan.py.

        A Step is the legacy representation of a step in the plan, and is still used in the
        Portia backend. If this step doesn't need to be represented in the plan sent to the Portia
        backend, return None.
        """
        raise NotImplementedError


# Test cases for PlanBuilderV2


def test_initialization_default_label() -> None:
    """Test PlanBuilderV2 initialization with default label."""
    builder = PlanBuilderV2()

    assert isinstance(builder.plan, PlanV2)
    assert builder.plan.label == "Run the plan built with the Plan Builder"
    assert builder.plan.steps == []
    assert builder.plan.plan_inputs == []
    assert builder.plan.summarize is False
    assert builder.plan.final_output_schema is None


def test_initialization_custom_label() -> None:
    """Test PlanBuilderV2 initialization with custom label."""
    custom_label = "Custom Plan Label"
    builder = PlanBuilderV2(label=custom_label)

    assert builder.plan.label == custom_label


def test_input_method() -> None:
    """Test the input() method for adding plan inputs."""
    builder = PlanBuilderV2()

    # Test adding input with name only
    result = builder.input(name="user_name")

    assert result is builder  # Should return self for chaining
    assert len(builder.plan.plan_inputs) == 1
    assert builder.plan.plan_inputs[0].name == "user_name"
    assert builder.plan.plan_inputs[0].description is None
    assert builder.plan.plan_inputs[0].value is None


def test_input_method_with_description() -> None:
    """Test the input() method with description."""
    builder = PlanBuilderV2()

    builder.input(name="user_name", description="The name of the user")

    assert len(builder.plan.plan_inputs) == 1
    assert builder.plan.plan_inputs[0].name == "user_name"
    assert builder.plan.plan_inputs[0].description == "The name of the user"
    assert builder.plan.plan_inputs[0].value is None


def test_input_method_multiple_inputs() -> None:
    """Test adding multiple inputs."""
    builder = PlanBuilderV2()

    builder.input(name="name", description="User name").input(name="age", description="User age")

    assert len(builder.plan.plan_inputs) == 2
    assert builder.plan.plan_inputs[0].name == "name"
    assert builder.plan.plan_inputs[1].name == "age"
    assert builder.plan.plan_inputs[0].value is None
    assert builder.plan.plan_inputs[1].value is None


def test_input_method_with_default_value() -> None:
    """Test the input() method with default value."""
    builder = PlanBuilderV2()

    builder.input(name="user_name", description="The name of the user", default_value="John Doe")

    assert len(builder.plan.plan_inputs) == 1
    assert builder.plan.plan_inputs[0].name == "user_name"
    assert builder.plan.plan_inputs[0].description == "The name of the user"
    assert builder.plan.plan_inputs[0].value == "John Doe"


def test_input_method_with_various_default_values() -> None:
    """Test the input() method with various types of default values."""
    builder = PlanBuilderV2()

    # Test with different types of default values
    default_bool = True
    builder.input(name="string_input", description="A string input", default_value="default_string")
    builder.input(name="int_input", description="An integer input", default_value=42)
    builder.input(name="bool_input", description="A boolean input", default_value=default_bool)
    builder.input(name="list_input", description="A list input", default_value=["item1", "item2"])
    builder.input(name="dict_input", description="A dict input", default_value={"key": "value"})
    builder.input(name="none_input", description="An input with explicit None", default_value=None)

    assert len(builder.plan.plan_inputs) == 6

    # Check each input's default value
    inputs = {inp.name: inp for inp in builder.plan.plan_inputs}
    assert inputs["string_input"].value == "default_string"
    assert inputs["int_input"].value == 42
    assert inputs["bool_input"].value is True
    assert inputs["list_input"].value == ["item1", "item2"]
    assert inputs["dict_input"].value == {"key": "value"}
    assert inputs["none_input"].value is None


def test_llm_step_method_basic() -> None:
    """Test the llm_step() method with basic parameters."""
    builder = PlanBuilderV2()

    result = builder.llm_step(task="Analyze the data")

    assert result is builder  # Should return self for chaining
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], LLMStep)
    assert builder.plan.steps[0].task == "Analyze the data"
    assert builder.plan.steps[0].inputs == []
    assert builder.plan.steps[0].output_schema is None
    assert builder.plan.steps[0].step_name == "step_0"


def test_llm_step_method_with_all_parameters() -> None:
    """Test the llm_step() method with all parameters."""
    builder = PlanBuilderV2()
    inputs = ["input1", StepOutput(0), Input("user_input")]

    builder.llm_step(
        task="Process the inputs",
        inputs=inputs,
        output_schema=OutputSchema,
        step_name="custom_step",
    )

    step = builder.plan.steps[0]
    assert isinstance(step, LLMStep)
    assert step.task == "Process the inputs"
    assert step.inputs == inputs
    assert step.output_schema == OutputSchema
    assert step.step_name == "custom_step"


def test_llm_step_method_with_model() -> None:
    """Test llm_step accepts a model parameter."""
    builder = PlanBuilderV2()
    builder.llm_step(task="Analyze", model="openai/gpt-4o")
    step = builder.plan.steps[0]
    assert isinstance(step, LLMStep)
    assert step.model == "openai/gpt-4o"


def test_llm_step_method_auto_generated_step_name() -> None:
    """Test that step names are auto-generated correctly."""
    builder = PlanBuilderV2()

    builder.llm_step(task="First step")
    builder.llm_step(task="Second step")

    assert builder.plan.steps[0].step_name == "step_0"
    assert builder.plan.steps[1].step_name == "step_1"


def test_invoke_tool_step_method_with_string_tool() -> None:
    """Test the invoke_tool_step() method with string tool identifier."""
    builder = PlanBuilderV2()
    args = {"param1": "value1", "param2": StepOutput(0)}

    result = builder.invoke_tool_step(tool="search_tool", args=args)

    assert result is builder  # Should return self for chaining
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], InvokeToolStep)
    assert builder.plan.steps[0].tool == "search_tool"
    assert builder.plan.steps[0].args == args
    assert builder.plan.steps[0].output_schema is None
    assert builder.plan.steps[0].step_name == "step_0"


def test_invoke_tool_step_method_with_tool_instance() -> None:
    """Test the invoke_tool_step() method with Tool instance."""
    builder = PlanBuilderV2()
    mock_tool = MockTool()

    builder.invoke_tool_step(tool=mock_tool, args={"input": "test"})

    step = builder.plan.steps[0]
    assert isinstance(step, InvokeToolStep)
    assert step.tool is mock_tool


def test_invoke_tool_step_method_with_all_parameters() -> None:
    """Test the invoke_tool_step() method with all parameters."""
    builder = PlanBuilderV2()

    builder.invoke_tool_step(
        tool="test_tool",
        args={"arg1": "value1"},
        output_schema=OutputSchema,
        step_name="tool_step",
    )

    step = builder.plan.steps[0]
    assert isinstance(step, InvokeToolStep)
    assert step.tool == "test_tool"
    assert step.args == {"arg1": "value1"}
    assert step.output_schema == OutputSchema
    assert step.step_name == "tool_step"


def test_invoke_tool_step_method_no_args() -> None:
    """Test the invoke_tool_step() method with no args."""
    builder = PlanBuilderV2()

    builder.invoke_tool_step(tool="no_args_tool")

    step = builder.plan.steps[0]
    assert isinstance(step, InvokeToolStep)
    assert step.args == {}


def test_function_step_method_basic() -> None:
    """Test the function_step() method with basic parameters."""
    builder = PlanBuilderV2()

    result = builder.function_step(function=example_function_for_testing)

    assert result is builder  # Should return self for chaining
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], InvokeToolStep)
    assert not isinstance(builder.plan.steps[0].tool, str)
    assert builder.plan.steps[0].tool.id == "local_function_example_function_for_testing"
    assert builder.plan.steps[0].tool.name == "Local Function Example Function For Testing"
    assert builder.plan.steps[0].args == {}
    assert builder.plan.steps[0].output_schema is None
    assert builder.plan.steps[0].step_name == "step_0"


def test_function_step_method_with_all_parameters() -> None:
    """Test the function_step() method with all parameters."""
    builder = PlanBuilderV2()
    args = {"x": 42, "y": Input("user_input")}

    builder.function_step(
        function=example_function_for_testing,
        args=args,
        output_schema=OutputSchema,
        step_name="func_step",
    )

    step = builder.plan.steps[0]
    assert isinstance(step, InvokeToolStep)
    assert not isinstance(step.tool, str)
    assert step.tool.id == "local_function_example_function_for_testing"
    assert step.tool.name == "Local Function Example Function For Testing"
    assert step.args == args
    assert step.output_schema == OutputSchema
    assert step.step_name == "func_step"


def test_single_tool_agent_step_method_basic() -> None:
    """Test the single_tool_agent_step() method with basic parameters."""
    builder = PlanBuilderV2()

    result = builder.single_tool_agent_step(tool="agent_tool", task="Complete the task")

    assert result is builder  # Should return self for chaining
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], SingleToolAgentStep)
    assert builder.plan.steps[0].tool == "agent_tool"
    assert builder.plan.steps[0].task == "Complete the task"
    assert builder.plan.steps[0].inputs == []
    assert builder.plan.steps[0].output_schema is None
    assert builder.plan.steps[0].step_name == "step_0"


def test_single_tool_agent_step_method_with_all_parameters() -> None:
    """Test the single_tool_agent_step() method with all parameters."""
    builder = PlanBuilderV2()
    inputs = ["context", StepOutput(0)]

    builder.single_tool_agent_step(
        tool="complex_tool",
        task="Process complex data",
        inputs=inputs,
        output_schema=OutputSchema,
        step_name="agent_step",
    )

    step = builder.plan.steps[0]
    assert isinstance(step, SingleToolAgentStep)
    assert step.tool == "complex_tool"
    assert step.task == "Process complex data"
    assert step.inputs == inputs
    assert step.output_schema == OutputSchema
    assert step.step_name == "agent_step"


def test_single_tool_agent_step_method_with_model() -> None:
    """Test single_tool_agent_step accepts a model parameter."""
    builder = PlanBuilderV2()
    builder.single_tool_agent_step(tool="search_tool", task="Find", model="openai/gpt-4o")
    step = builder.plan.steps[0]
    assert isinstance(step, SingleToolAgentStep)
    assert step.model == "openai/gpt-4o"


def test_single_tool_agent_step_accepts_tool_object() -> None:
    """Test single_tool_agent_step accepts a Tool instance."""
    builder = PlanBuilderV2()
    mock_tool = MockTool()

    builder.single_tool_agent_step(tool=mock_tool, task="Use the tool")

    step = builder.plan.steps[0]
    assert isinstance(step, SingleToolAgentStep)
    assert step.tool is mock_tool


def test_react_agent_step_method_basic() -> None:
    """Test the react_agent_step() method with basic parameters."""
    builder = PlanBuilderV2()
    tools = ["search_tool", "calculator_tool"]

    result = builder.react_agent_step(task="Research and calculate", tools=tools)

    assert result is builder  # Should return self for chaining
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], ReActAgentStep)
    assert builder.plan.steps[0].task == "Research and calculate"
    assert builder.plan.steps[0].tools == tools
    assert builder.plan.steps[0].inputs == []
    assert builder.plan.steps[0].output_schema is None
    assert builder.plan.steps[0].step_name == "step_0"
    assert builder.plan.steps[0].tool_call_limit == 25
    assert builder.plan.steps[0].allow_agent_clarifications is False


def test_react_agent_step_accepts_tool_objects() -> None:
    """Test react_agent_step accepts Tool instances."""
    builder = PlanBuilderV2()
    tools = [MockTool()]

    builder.react_agent_step(task="Research", tools=tools)

    step = builder.plan.steps[0]
    assert isinstance(step, ReActAgentStep)
    assert step.tools == tools


def test_react_agent_step_method_with_all_parameters() -> None:
    """Test the react_agent_step() method with all parameters."""
    builder = PlanBuilderV2()
    tools = ["search_tool", "calculator_tool", "weather_tool"]
    inputs = ["context", StepOutput(0), Input("user_query")]

    builder.react_agent_step(
        task="Complex multi-tool analysis",
        tools=tools,
        inputs=inputs,
        output_schema=OutputSchema,
        step_name="react_analysis",
        allow_agent_clarifications=True,
        tool_call_limit=50,
    )

    step = builder.plan.steps[0]
    assert isinstance(step, ReActAgentStep)
    assert step.task == "Complex multi-tool analysis"
    assert step.tools == tools
    assert step.inputs == inputs
    assert step.output_schema == OutputSchema
    assert step.step_name == "react_analysis"
    assert step.allow_agent_clarifications is True
    assert step.tool_call_limit == 50


def test_react_agent_step_method_with_model() -> None:
    """Test react_agent_step accepts a model parameter."""
    builder = PlanBuilderV2()
    builder.react_agent_step(task="Research", model="openai/gpt-4o")
    step = builder.plan.steps[0]
    assert isinstance(step, ReActAgentStep)
    assert step.model == "openai/gpt-4o"


def test_react_agent_step_method_single_tool() -> None:
    """Test the react_agent_step() method with a single tool."""
    builder = PlanBuilderV2()

    builder.react_agent_step(task="Simple task", tools=["single_tool"])

    step = builder.plan.steps[0]
    assert isinstance(step, ReActAgentStep)
    assert step.tools == ["single_tool"]


def test_user_verify_step_method() -> None:
    """Test the user_verify_step() method."""
    builder = PlanBuilderV2()

    result = builder.user_verify(message="Check this")

    assert result is builder
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], UserVerifyStep)
    assert builder.plan.steps[0].message == "Check this"
    assert builder.plan.steps[0].step_name == "step_0"


def test_user_input_text_input_method() -> None:
    """Test the user_input() method for text input."""
    builder = PlanBuilderV2()

    result = builder.user_input(message="Please enter your name")

    assert result is builder
    assert len(builder.plan.steps) == 1
    assert isinstance(builder.plan.steps[0], UserInputStep)
    assert builder.plan.steps[0].message == "Please enter your name"
    assert builder.plan.steps[0].options is None
    assert builder.plan.steps[0].step_name == "step_0"


def test_user_input_multiple_choice_method() -> None:
    """Test the user_input() method for multiple choice."""
    builder = PlanBuilderV2()
    options = ["red", "green", "blue"]

    result = builder.user_input(
        message="Choose your favorite color",
        options=options,
        step_name="choose_color",
    )

    assert result is builder
    assert len(builder.plan.steps) == 1
    step = builder.plan.steps[0]
    assert isinstance(step, UserInputStep)
    assert step.message == "Choose your favorite color"
    assert step.options == options
    assert step.step_name == "choose_color"


def test_final_output_method_basic() -> None:
    """Test the final_output() method with basic parameters."""
    builder = PlanBuilderV2()

    result = builder.final_output()

    assert result is builder  # Should return self for chaining
    assert builder.plan.final_output_schema is None
    assert builder.plan.summarize is False


def test_final_output_method_with_schema() -> None:
    """Test the final_output() method with output schema."""
    builder = PlanBuilderV2()

    builder.final_output(output_schema=OutputSchema)

    assert builder.plan.final_output_schema == OutputSchema
    assert builder.plan.summarize is False


def test_final_output_method_with_summarize() -> None:
    """Test the final_output() method with summarize enabled."""
    builder = PlanBuilderV2()

    builder.final_output(summarize=True)

    assert builder.plan.final_output_schema is None
    assert builder.plan.summarize is True


def test_final_output_method_with_all_parameters() -> None:
    """Test the final_output() method with all parameters."""
    builder = PlanBuilderV2()

    builder.final_output(output_schema=OutputSchema, summarize=True)

    assert builder.plan.final_output_schema == OutputSchema
    assert builder.plan.summarize is True


def test_build_method() -> None:
    """Test the build() method returns correct PlanV2 instance."""
    builder = PlanBuilderV2(label="Test Plan")

    builder.input(name="test_input", description="Test input description")
    builder.llm_step(task="Test task")
    builder.final_output(output_schema=OutputSchema, summarize=True)

    plan = builder.build()

    assert isinstance(plan, PlanV2)
    assert plan is builder.plan  # Should return the same instance
    assert plan.label == "Test Plan"
    assert len(plan.plan_inputs) == 1
    assert len(plan.steps) == 1
    assert plan.final_output_schema == OutputSchema
    assert plan.summarize is True


def test_method_chaining() -> None:
    """Test that all methods return self for proper chaining."""
    builder = PlanBuilderV2("Chaining Test")

    result = (
        builder.input(name="user_name", description="Name of the user", default_value="John Doe")
        .input(name="user_age", description="Age of the user", default_value=25)
        .llm_step(task="Analyze user info", inputs=[Input("user_name"), Input("user_age")])
        .invoke_tool_step(tool="search_tool", args={"query": StepOutput(0)})
        .function_step(function=example_function_for_testing, args={"x": 1, "y": "test"})
        .single_tool_agent_step(tool="agent_tool", task="Final processing")
        .final_output(output_schema=OutputSchema, summarize=True)
    )

    assert result is builder

    # Verify the plan was built correctly
    plan = builder.build()
    assert len(plan.plan_inputs) == 2
    assert len(plan.steps) == 4
    assert isinstance(plan.steps[0], LLMStep)
    assert isinstance(plan.steps[1], InvokeToolStep)
    assert isinstance(plan.steps[2], InvokeToolStep)
    assert isinstance(plan.steps[3], SingleToolAgentStep)
    assert plan.final_output_schema == OutputSchema
    assert plan.summarize is True

    # Verify default values are set correctly
    inputs = {inp.name: inp for inp in plan.plan_inputs}
    assert inputs["user_name"].value == "John Doe"
    assert inputs["user_age"].value == 25


def test_empty_plan_build() -> None:
    """Test building an empty plan."""
    builder = PlanBuilderV2()
    plan = builder.build()

    assert isinstance(plan, PlanV2)
    assert len(plan.steps) == 0
    assert len(plan.plan_inputs) == 0
    assert plan.final_output_schema is None
    assert plan.summarize is False


def test_step_name_generation_with_mixed_steps() -> None:
    """Test step name generation with different types of steps."""
    builder = PlanBuilderV2()

    builder.llm_step(task="LLM task")
    builder.invoke_tool_step(tool="tool1")
    builder.function_step(function=example_function_for_testing)
    builder.single_tool_agent_step(tool="agent_tool", task="Agent task")

    assert builder.plan.steps[0].step_name == "step_0"
    assert builder.plan.steps[1].step_name == "step_1"
    assert builder.plan.steps[2].step_name == "step_2"
    assert builder.plan.steps[3].step_name == "step_3"


def test_custom_step_names_override_auto_generation() -> None:
    """Test that custom step names override auto-generation."""
    builder = PlanBuilderV2()

    builder.llm_step(task="First", step_name="custom_first")
    builder.llm_step(task="Second")  # Should get step_1
    builder.llm_step(task="Third", step_name="custom_third")

    assert builder.plan.steps[0].step_name == "custom_first"
    assert builder.plan.steps[1].step_name == "step_1"
    assert builder.plan.steps[2].step_name == "custom_third"


def test_references_in_inputs_and_args() -> None:
    """Test using references (StepOutput and Input) in various contexts."""
    builder = PlanBuilderV2()

    # Add inputs to reference
    builder.input(name="user_query", description="The user's query")

    # Add steps with references
    builder.llm_step(task="Process query", inputs=[Input("user_query"), "additional context"])
    builder.invoke_tool_step(tool="search_tool", args={"query": StepOutput(0), "limit": 10})
    builder.function_step(function=example_function_for_testing, args={"x": 42, "y": StepOutput(1)})

    plan = builder.build()

    # Verify references are preserved
    llm_step = plan.steps[0]
    assert isinstance(llm_step, LLMStep)
    assert isinstance(llm_step.inputs[0], Input)
    assert llm_step.inputs[0].name == "user_query"

    tool_step = plan.steps[1]
    assert isinstance(tool_step, InvokeToolStep)
    assert isinstance(tool_step.args["query"], StepOutput)
    assert tool_step.args["query"].step == 0

    func_step = plan.steps[2]
    assert isinstance(func_step, InvokeToolStep)
    assert isinstance(func_step.args["y"], StepOutput)
    assert func_step.args["y"].step == 1


def test_add_step_method_basic() -> None:
    """Test the add_step() method with basic functionality."""
    builder = PlanBuilderV2()
    custom_step = CustomStep(step_name="custom_step")

    result = builder.add_step(custom_step)

    assert len(result.plan.steps) == 1
    assert result.plan.steps[0] is custom_step
    assert result.plan.steps[0].step_name == "custom_step"


def test_add_step_method_with_different_step_types() -> None:
    """Test the add_step() method with different step types."""
    builder = PlanBuilderV2()

    llm_step = LLMStep(task="LLM task", step_name="llm_step")
    tool_step = InvokeToolStep(tool="search_tool", args={"query": "test"}, step_name="tool_step")
    func_step = InvokeToolStep(
        tool=tool(example_function_for_testing)(),
        args={"x": 1, "y": "test"},
        step_name="func_step",
    )
    agent_step = SingleToolAgentStep(tool="agent_tool", task="Agent task", step_name="agent_step")

    builder.add_step(llm_step).add_step(tool_step).add_step(func_step).add_step(agent_step)

    assert len(builder.plan.steps) == 4
    assert isinstance(builder.plan.steps[0], LLMStep)
    assert isinstance(builder.plan.steps[1], InvokeToolStep)
    assert isinstance(builder.plan.steps[2], InvokeToolStep)
    assert isinstance(builder.plan.steps[3], SingleToolAgentStep)


def test_add_steps_method_with_iterable() -> None:
    """Test the add_steps() method with an iterable of steps."""
    builder = PlanBuilderV2()

    steps = [
        LLMStep(task="First task", step_name="step1"),
        InvokeToolStep(tool="test_tool", args={"input": "test"}, step_name="step2"),
        InvokeToolStep(
            tool=tool(example_function_for_testing)(),
            args={"x": 42, "y": "hello"},
            step_name="step3",
        ),
    ]

    result = builder.add_steps(steps)

    assert len(result.plan.steps) == 3
    assert result.plan.steps[0] is steps[0]
    assert result.plan.steps[1] is steps[1]
    assert result.plan.steps[2] is steps[2]


def test_add_steps_method_with_plan_v2() -> None:
    """Test the add_steps() method with a PlanV2 instance."""
    builder = PlanBuilderV2()

    other_plan = PlanV2(
        label="Other plan",
        steps=[
            LLMStep(task="Task from other plan", step_name="other_step1"),
            InvokeToolStep(tool="other_tool", args={}, step_name="other_step2"),
        ],
        plan_inputs=[
            PlanInput(name="other_input", description="Input from other plan"),
            PlanInput(name="another_input", description="Another input", value="default"),
        ],
    )

    result = builder.input(name="input1").add_steps(other_plan)

    assert len(result.plan.steps) == 2
    assert len(result.plan.plan_inputs) == 3

    assert result.plan.steps[0] is other_plan.steps[0]
    assert result.plan.steps[1] is other_plan.steps[1]

    assert result.plan.plan_inputs[0].name == "input1"
    assert result.plan.plan_inputs[1].name == "other_input"
    assert result.plan.plan_inputs[2].name == "another_input"


def test_add_steps_method_with_plan_v2_duplicate_inputs_error() -> None:
    """Test the add_steps() method raises error for duplicate plan inputs."""
    builder = PlanBuilderV2()

    builder.input(name="shared_input", description="Shared input name")

    other_plan = PlanV2(
        label="Other plan",
        steps=[LLMStep(task="Task", step_name="step1")],
        plan_inputs=[
            PlanInput(name="shared_input", description="Duplicate input name"),
            PlanInput(name="unique_input", description="Unique input"),
        ],
    )

    # Should raise PlanBuilderError due to duplicate input
    with pytest.raises(PlanBuilderError, match="Duplicate input shared_input found in plan"):
        builder.add_steps(other_plan)


def test_add_steps_method_with_empty_iterable() -> None:
    """Test the add_steps() method with an empty iterable."""
    builder = PlanBuilderV2().llm_step(task="Initial step").add_steps([])
    assert len(builder.plan.steps) == 1


def test_add_steps_method_with_empty_plan_v2() -> None:
    """Test the add_steps() method with an empty PlanV2."""
    builder = PlanBuilderV2().input(name="existing_input").llm_step(task="Initial step")
    empty_plan = PlanBuilderV2().build()

    result = builder.add_steps(empty_plan)

    assert len(result.plan.steps) == 1
    assert len(result.plan.plan_inputs) == 1


def test_add_steps_method_chaining_with_different_sources() -> None:
    """Test chaining add_steps() method with different sources."""
    builder = PlanBuilderV2()

    step_list = [LLMStep(task="List step", step_name="list_step")]
    plan_with_steps = PlanV2(
        label="Source plan",
        steps=[InvokeToolStep(tool="plan_tool", args={}, step_name="plan_step")],
        plan_inputs=[PlanInput(name="plan_input", description="From plan")],
    )

    result = builder.add_steps(step_list).add_steps(plan_with_steps)

    assert len(result.plan.steps) == 2
    assert len(result.plan.plan_inputs) == 1
    assert result.plan.steps[0].step_name == "list_step"
    assert result.plan.steps[1].step_name == "plan_step"
    assert result.plan.plan_inputs[0].name == "plan_input"


def test_add_step_and_add_steps_integration() -> None:
    """Test integration of add_step and add_steps methods together."""
    step_batch = (
        PlanBuilderV2()
        .invoke_tool_step(tool="batch_tool", args={}, step_name="batch1")
        .function_step(function=example_function_for_testing, args={}, step_name="batch2")
        .build()
    )

    builder = (
        PlanBuilderV2()
        .add_step(LLMStep(task="Individual step", step_name="individual"))
        .add_steps(step_batch)
        .add_steps(
            PlanBuilderV2().add_step(LLMStep(task="From plan", step_name="from_plan")).build()
        )
        .add_step(SingleToolAgentStep(tool="final_tool", task="Final step", step_name="final"))
    )

    assert len(builder.plan.steps) == 5
    assert len(builder.plan.plan_inputs) == 0
    assert builder.plan.steps[0].step_name == "individual"
    assert builder.plan.steps[1].step_name == "batch1"
    assert builder.plan.steps[2].step_name == "batch2"
    assert builder.plan.steps[3].step_name == "from_plan"
    assert builder.plan.steps[4].step_name == "final"


def test_on_error_attaches_handler_and_ignore_errors_sets_none() -> None:
    """Ensure on_error attaches to previous step and ignore_errors sets None handler."""
    builder = (
        PlanBuilderV2()
        .function_step(function=lambda: "ok", step_name="will_be_overridden")
        .on_error(lambda e: "handled")
    )
    assert callable(builder.plan.steps[-1].on_error)

    builder.ignore_errors()
    assert callable(builder.plan.steps[-1].on_error)
    # Call to verify lambda signature works
    assert builder.plan.steps[-1].on_error(Exception("x")) is None  # type: ignore[union-attr]


def test_on_error_without_previous_step_raises() -> None:
    """Calling on_error before adding any steps should raise PlanBuilderError."""
    with pytest.raises(PlanBuilderError, match="on_error must be called after adding a step"):
        PlanBuilderV2().on_error(lambda e: None)


def test_basic_if_endif_block() -> None:
    """Test basic if-endif conditional block."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return True

    result = builder.if_(test_condition).llm_step(task="Inside if block").endif()

    assert result is builder
    assert len(builder.plan.steps) == 3

    # Check if step
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)
    assert if_step.condition is test_condition
    assert if_step.args == {}
    assert if_step.block_clause_type == ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK
    assert if_step.clause_index_in_block == 0
    assert if_step.step_name == "step_0"

    # Check LLM step inside block
    llm_step = builder.plan.steps[1]
    assert isinstance(llm_step, LLMStep)
    assert llm_step.task == "Inside if block"
    assert llm_step.conditional_block is not None
    assert llm_step.conditional_block is if_step.conditional_block

    # Check endif step
    endif_step = builder.plan.steps[2]
    assert isinstance(endif_step, ConditionalStep)
    assert endif_step.block_clause_type == ConditionalBlockClauseType.END_CONDITION_BLOCK
    assert endif_step.clause_index_in_block == 1
    assert endif_step.conditional_block is if_step.conditional_block


def test_if_else_endif_block() -> None:
    """Test if-else-endif conditional block."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return False

    result = (
        builder.if_(test_condition)
        .llm_step(task="Inside if block")
        .else_()
        .llm_step(task="Inside else block")
        .endif()
    )

    assert result is builder
    assert len(builder.plan.steps) == 5

    # Check if step
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)
    assert if_step.block_clause_type == ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK

    # Check else step
    else_step = builder.plan.steps[2]
    assert isinstance(else_step, ConditionalStep)
    assert else_step.block_clause_type == ConditionalBlockClauseType.ALTERNATE_CLAUSE
    assert else_step.clause_index_in_block == 1
    assert else_step.conditional_block is if_step.conditional_block

    # Check endif step
    endif_step = builder.plan.steps[4]
    assert isinstance(endif_step, ConditionalStep)
    assert endif_step.block_clause_type == ConditionalBlockClauseType.END_CONDITION_BLOCK


def test_if_else_if_else_endif_block() -> None:
    """Test if-else_if-else-endif conditional block."""
    builder = PlanBuilderV2()

    def first_condition() -> bool:
        return False

    def second_condition() -> bool:
        return True

    result = (
        builder.if_(first_condition)
        .llm_step(task="Inside if block")
        .else_if_(second_condition)
        .llm_step(task="Inside else_if block")
        .else_()
        .llm_step(task="Inside else block")
        .endif()
    )

    assert result is builder
    assert len(builder.plan.steps) == 7

    # Check if step
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)
    assert if_step.condition is first_condition
    assert if_step.block_clause_type == ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK
    assert if_step.clause_index_in_block == 0

    # Check else_if step
    elif_step = builder.plan.steps[2]
    assert isinstance(elif_step, ConditionalStep)
    assert elif_step.condition is second_condition
    assert elif_step.block_clause_type == ConditionalBlockClauseType.ALTERNATE_CLAUSE
    assert elif_step.clause_index_in_block == 1
    assert elif_step.conditional_block is if_step.conditional_block

    # Check else step
    else_step = builder.plan.steps[4]
    assert isinstance(else_step, ConditionalStep)
    assert else_step.block_clause_type == ConditionalBlockClauseType.ALTERNATE_CLAUSE
    assert else_step.clause_index_in_block == 2
    assert else_step.conditional_block is if_step.conditional_block

    # Check endif step
    endif_step = builder.plan.steps[6]
    assert isinstance(endif_step, ConditionalStep)
    assert endif_step.block_clause_type == ConditionalBlockClauseType.END_CONDITION_BLOCK
    assert endif_step.clause_index_in_block == 3
    assert endif_step.conditional_block is if_step.conditional_block


def test_nested_if_blocks() -> None:
    """Test nested if blocks."""
    builder = PlanBuilderV2()

    def outer_condition() -> bool:
        return True

    def inner_condition() -> bool:
        return False

    result = (
        builder.if_(outer_condition)
        .llm_step(task="Outer if block")
        .if_(inner_condition)
        .llm_step(task="Inner if block")
        .else_()
        .llm_step(task="Inner else block")
        .endif()
        .llm_step(task="Back to outer if")
        .endif()
    )

    assert result is builder
    assert len(builder.plan.steps) == 9

    # Check outer if
    outer_if = builder.plan.steps[0]
    assert isinstance(outer_if, ConditionalStep)
    assert outer_if.condition is outer_condition
    assert outer_if.conditional_block.parent_conditional_block is None  # pyright: ignore[reportOptionalMemberAccess]

    # Check inner if
    inner_if = builder.plan.steps[2]
    assert isinstance(inner_if, ConditionalStep)
    assert inner_if.condition is inner_condition
    assert inner_if.conditional_block.parent_conditional_block is outer_if.conditional_block  # pyright: ignore[reportOptionalMemberAccess]

    # Check inner else
    inner_else = builder.plan.steps[4]
    assert isinstance(inner_else, ConditionalStep)
    assert inner_else.conditional_block is inner_if.conditional_block

    # Check inner endif
    inner_endif = builder.plan.steps[6]
    assert isinstance(inner_endif, ConditionalStep)
    assert inner_endif.conditional_block is inner_if.conditional_block

    # Check outer endif
    outer_endif = builder.plan.steps[8]
    assert isinstance(outer_endif, ConditionalStep)
    assert outer_endif.conditional_block is outer_if.conditional_block


def test_conditional_with_string_condition() -> None:
    """Test conditional block with string condition."""
    builder = PlanBuilderV2()

    result = builder.if_("len(input_data) > 0").llm_step(task="Process data").endif()

    assert result is builder
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)
    assert if_step.condition == "len(input_data) > 0"


def test_conditional_with_args() -> None:
    """Test conditional block with arguments."""
    builder = PlanBuilderV2()

    def test_condition(value: int) -> bool:
        return value > 10

    args = {"value": Input("user_input")}
    result = builder.if_(test_condition, args).llm_step(task="High value processing").endif()

    assert result is builder
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)
    assert if_step.condition is test_condition
    assert if_step.args == args


def test_else_if_without_if_raises_error() -> None:
    """Test that else_if_ raises error when called without if_."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return True

    with pytest.raises(PlanBuilderError, match="else_if_ must be called from a conditional block"):
        builder.else_if_(test_condition)


def test_else_without_if_raises_error() -> None:
    """Test that else_ raises error when called without if_."""
    builder = PlanBuilderV2()

    with pytest.raises(PlanBuilderError, match="else_ must be called from a conditional block"):
        builder.else_()


def test_endif_without_if_raises_error() -> None:
    """Test that endif raises error when called without if_."""
    builder = PlanBuilderV2()

    with pytest.raises(PlanBuilderError, match="endif must be called from a conditional block"):
        builder.endif()


def test_build_with_missing_endif_raises_error() -> None:
    """Test that build() raises error when endif is missing."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return True

    builder.if_(test_condition).llm_step(task="Inside if block")
    # Missing endif()

    with pytest.raises(PlanBuilderError, match="Please add an endif or endloop for all blocks."):
        builder.build()


def test_conditional_method_chaining() -> None:
    """Test that conditional methods return self for chaining."""
    builder = PlanBuilderV2()

    def condition1() -> bool:
        return True

    def condition2() -> bool:
        return False

    result = (
        builder.input(name="test_input")
        .if_(condition1)
        .llm_step(task="First branch")
        .else_if_(condition2)
        .invoke_tool_step(tool="test_tool", args={"query": Input("test_input")})
        .else_()
        .function_step(function=example_function_for_testing, args={"x": 1, "y": "test"})
        .endif()
        .llm_step(task="After conditional")
    )

    assert result is builder
    plan = builder.build()
    assert len(plan.steps) == 8  # if, llm, else_if, tool, else, func, endif, final_llm
    assert len(plan.plan_inputs) == 1


def test_add_steps_with_input_values_single_value() -> None:
    """Test add_steps with input_values setting a single input value."""
    builder = PlanBuilderV2()

    sub_plan = (
        PlanBuilderV2()
        .input(name="sub_input", description="Input for sub plan")
        .llm_step(task="Sub task", step_name="sub_step")
        .build()
    )

    result = builder.add_steps(sub_plan, input_values={"sub_input": "provided_value"})

    assert len(result.plan.plan_inputs) == 1
    assert result.plan.plan_inputs[0].name == "sub_input"
    assert result.plan.plan_inputs[0].value == "provided_value"


def test_add_steps_with_input_values_multiple_values_override_default() -> None:
    """Test add_steps with input_values setting 2 out of 4 values, overriding default."""
    builder = PlanBuilderV2()

    sub_plan = (
        PlanBuilderV2()
        .input(name="input_no_default", description="Input without default")
        .input(
            name="input_with_default",
            description="Input with default",
            default_value="original_default",
        )
        .input(name="input_unchanged1", description="Unchanged input 1")
        .input(
            name="input_unchanged2",
            description="Unchanged input 2",
            default_value="unchanged_default",
        )
        .llm_step(task="Sub task", step_name="sub_step")
        .build()
    )

    result = builder.add_steps(
        sub_plan,
        input_values={
            "input_no_default": "new_value_no_default",
            "input_with_default": "overridden_value",
        },
    )

    assert len(result.plan.plan_inputs) == 4

    inputs = {inp.name: inp for inp in result.plan.plan_inputs}
    assert inputs["input_no_default"].value == "new_value_no_default"
    assert inputs["input_with_default"].value == "overridden_value"  # Should override default
    assert inputs["input_unchanged1"].value is None
    assert inputs["input_unchanged2"].value == "unchanged_default"


def test_add_steps_with_input_values_invalid_input_name_error() -> None:
    """Test add_steps raises error when input_values contains invalid input name."""
    builder = PlanBuilderV2()

    sub_plan = (
        PlanBuilderV2()
        .input(name="valid_input", description="Valid input")
        .llm_step(task="Sub task", step_name="sub_step")
        .build()
    )

    with pytest.raises(PlanBuilderError):
        builder.add_steps(sub_plan, input_values={"invalid_input": "some_value"})


# Loop tests


def test_basic_loop_with_condition() -> None:
    """Test basic loop with condition."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return True

    builder.loop(while_=test_condition).llm_step(task="Test step").end_loop()

    plan = builder.build()
    assert len(plan.steps) == 3
    assert isinstance(plan.steps[0], LoopStep)
    assert plan.steps[0].loop_type == LoopType.WHILE
    assert plan.steps[0].loop_step_type == LoopStepType.START


def test_basic_loop_over_condition() -> None:
    """Test basic loop with condition."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return True

    result = builder.loop(while_=test_condition).llm_step(task="Inside loop").end_loop()

    assert result is builder
    assert len(builder.plan.steps) == 3

    # Check loop step
    loop_step = builder.plan.steps[0]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.condition is test_condition
    assert loop_step.over is None
    assert loop_step.loop_type == LoopType.WHILE
    assert loop_step.loop_step_type == LoopStepType.START
    assert loop_step.step_name == "step_0"

    # Check LLM step inside loop
    llm_step = builder.plan.steps[1]
    assert isinstance(llm_step, LLMStep)
    assert llm_step.task == "Inside loop"

    # Check end_loop step
    end_loop_step = builder.plan.steps[2]
    assert isinstance(end_loop_step, LoopStep)
    assert end_loop_step.loop_step_type == LoopStepType.END
    assert end_loop_step.loop_type == LoopType.WHILE
    assert end_loop_step.condition is test_condition


def test_basic_loop_over_reference() -> None:
    """Test basic loop with over reference."""
    builder = PlanBuilderV2()

    result = (
        builder.function_step(function=lambda: [1, 2, 3], step_name="items")
        .loop(over=StepOutput("items"))
        .llm_step(task="Process item")
        .end_loop()
    )

    assert result is builder
    assert len(builder.plan.steps) == 4

    # Check function step
    func_step = builder.plan.steps[0]
    assert isinstance(func_step, InvokeToolStep)
    assert func_step.step_name == "items"

    # Check loop step
    loop_step = builder.plan.steps[1]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.over == StepOutput("items")
    assert loop_step.condition is None
    assert loop_step.loop_type == LoopType.FOR_EACH
    assert loop_step.loop_step_type == LoopStepType.START

    # Check LLM step inside loop
    llm_step = builder.plan.steps[2]
    assert isinstance(llm_step, LLMStep)
    assert llm_step.task == "Process item"

    # Check end_loop step
    end_loop_step = builder.plan.steps[3]
    assert isinstance(end_loop_step, LoopStep)
    assert end_loop_step.loop_step_type == LoopStepType.END
    assert end_loop_step.loop_type == LoopType.FOR_EACH


def test_loop_with_args() -> None:
    """Test loop with args parameter."""
    builder = PlanBuilderV2()

    def test_condition(x: int) -> bool:
        return x > 0

    result = (
        builder.loop(while_=test_condition, args={"x": 5}).llm_step(task="Inside loop").end_loop()
    )

    assert result is builder
    assert len(builder.plan.steps) == 3

    # Check loop step
    loop_step = builder.plan.steps[0]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.condition is test_condition
    assert loop_step.args == {"x": 5}

    # Check end_loop step
    end_loop_step = builder.plan.steps[2]
    assert isinstance(end_loop_step, LoopStep)
    assert end_loop_step.args == {"x": 5}


def test_loop_with_custom_step_name() -> None:
    """Test loop with custom step names."""
    builder = PlanBuilderV2()

    result = (
        builder.loop(while_=lambda: True, step_name="custom_loop")
        .llm_step(task="Inside loop")
        .end_loop(step_name="custom_end_loop")
    )

    assert result is builder
    assert len(builder.plan.steps) == 3

    # Check loop step
    loop_step = builder.plan.steps[0]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.step_name == "custom_loop"

    # Check end_loop step
    end_loop_step = builder.plan.steps[2]
    assert isinstance(end_loop_step, LoopStep)
    assert end_loop_step.step_name == "custom_end_loop"


def test_nested_loops() -> None:
    """Test nested loops."""
    builder = PlanBuilderV2()

    result = (
        builder.loop(while_=lambda: True, step_name="outer_loop")
        .llm_step(task="Outer loop")
        .loop(over=StepOutput("items"), step_name="inner_loop")
        .llm_step(task="Inner loop")
        .end_loop(step_name="inner_end")
        .llm_step(task="Back to outer")
        .end_loop(step_name="outer_end")
    )

    assert result is builder
    assert len(builder.plan.steps) == 7

    # Check outer loop
    outer_loop = builder.plan.steps[0]
    assert isinstance(outer_loop, LoopStep)
    assert outer_loop.step_name == "outer_loop"
    assert outer_loop.loop_type == LoopType.WHILE

    # Check inner loop
    inner_loop = builder.plan.steps[2]
    assert isinstance(inner_loop, LoopStep)
    assert inner_loop.step_name == "inner_loop"
    assert inner_loop.loop_type == LoopType.FOR_EACH

    # Check inner end_loop
    inner_end = builder.plan.steps[4]
    assert isinstance(inner_end, LoopStep)
    assert inner_end.step_name == "inner_end"

    # Check outer end_loop
    outer_end = builder.plan.steps[6]
    assert isinstance(outer_end, LoopStep)
    assert outer_end.step_name == "outer_end"


def test_loop_inside_if_block() -> None:
    """Test loop inside if block."""
    builder = PlanBuilderV2()

    def test_condition() -> bool:
        return True

    result = (
        builder.if_(test_condition)
        .loop(while_=lambda: True)
        .llm_step(task="Inside loop in if")
        .end_loop()
        .endif()
    )

    assert result is builder
    assert len(builder.plan.steps) == 5

    # Check if step
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)

    # Check loop step
    loop_step = builder.plan.steps[1]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.loop_type == LoopType.WHILE

    # Check LLM step inside loop
    llm_step = builder.plan.steps[2]
    assert isinstance(llm_step, LLMStep)
    assert llm_step.task == "Inside loop in if"

    # Check end_loop step
    end_loop_step = builder.plan.steps[3]
    assert isinstance(end_loop_step, LoopStep)

    # Check endif step
    endif_step = builder.plan.steps[4]
    assert isinstance(endif_step, ConditionalStep)


def test_if_inside_loop_block() -> None:
    """Test if block inside loop."""
    builder = PlanBuilderV2()

    def loop_condition() -> bool:
        return True

    def if_condition() -> bool:
        return True

    result = (
        builder.loop(while_=loop_condition)
        .if_(if_condition)
        .llm_step(task="Inside if in loop")
        .else_()
        .llm_step(task="Inside else in loop")
        .endif()
        .end_loop()
    )

    assert result is builder
    assert len(builder.plan.steps) == 7

    # Check loop step
    loop_step = builder.plan.steps[0]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.loop_type == LoopType.WHILE

    # Check if step
    if_step = builder.plan.steps[1]
    assert isinstance(if_step, ConditionalStep)

    # Check else step
    else_step = builder.plan.steps[3]
    assert isinstance(else_step, ConditionalStep)

    # Check endif step
    endif_step = builder.plan.steps[5]
    assert isinstance(endif_step, ConditionalStep)

    # Check end_loop step
    end_loop_step = builder.plan.steps[6]
    assert isinstance(end_loop_step, LoopStep)


def test_loop_inside_else_block() -> None:
    """Test loop inside else block."""
    builder = PlanBuilderV2()

    def if_condition() -> bool:
        return False

    result = (
        builder.if_(if_condition)
        .llm_step(task="Inside if")
        .else_()
        .loop(over=StepOutput("items"))
        .llm_step(task="Inside loop in else")
        .end_loop()
        .endif()
    )

    assert result is builder
    assert len(builder.plan.steps) == 7

    # Check if step
    if_step = builder.plan.steps[0]
    assert isinstance(if_step, ConditionalStep)

    # Check else step
    else_step = builder.plan.steps[2]
    assert isinstance(else_step, ConditionalStep)

    # Check loop step
    loop_step = builder.plan.steps[3]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.loop_type == LoopType.FOR_EACH

    # Check end_loop step
    end_loop_step = builder.plan.steps[5]
    assert isinstance(end_loop_step, LoopStep)

    # Check endif step
    endif_step = builder.plan.steps[6]
    assert isinstance(endif_step, ConditionalStep)


def test_loop_inside_else_if_block() -> None:
    """Test loop inside else_if block."""
    builder = PlanBuilderV2()

    def first_condition() -> bool:
        return False

    def second_condition() -> bool:
        return True

    result = (
        builder.if_(first_condition)
        .llm_step(task="Inside first if")
        .else_if_(second_condition)
        .loop(while_=lambda: True)
        .llm_step(task="Inside loop in else_if")
        .end_loop()
        .else_()
        .llm_step(task="Inside else")
        .endif()
    )

    assert result is builder
    assert len(builder.plan.steps) == 9

    # Check first if step
    first_if = builder.plan.steps[0]
    assert isinstance(first_if, ConditionalStep)

    # Check else_if step
    else_if = builder.plan.steps[2]
    assert isinstance(else_if, ConditionalStep)

    # Check loop step
    loop_step = builder.plan.steps[3]
    assert isinstance(loop_step, LoopStep)

    # Check end_loop step
    end_loop_step = builder.plan.steps[5]
    assert isinstance(end_loop_step, LoopStep)

    # Check else step
    else_step = builder.plan.steps[6]
    assert isinstance(else_step, ConditionalStep)

    # Check endif step
    endif_step = builder.plan.steps[8]
    assert isinstance(endif_step, ConditionalStep)


def test_loop_with_string_condition() -> None:
    """Test loop with string condition."""
    builder = PlanBuilderV2()

    result = builder.loop(while_="x > 0", args={"x": 5}).llm_step(task="Inside loop").end_loop()

    assert result is builder
    assert len(builder.plan.steps) == 3

    # Check loop step
    loop_step = builder.plan.steps[0]
    assert isinstance(loop_step, LoopStep)
    assert loop_step.condition == "x > 0"
    assert loop_step.args == {"x": 5}


def test_loop_error_condition_and_over_both_set() -> None:
    """Test that loop raises error when both condition and over are set."""
    builder = PlanBuilderV2()

    with pytest.raises(PlanBuilderError, match="Only one of while_, do_while_, or over can be set"):
        builder.loop(while_=lambda: True, over=StepOutput("items"))


def test_loop_error_neither_condition_nor_over_set() -> None:
    """Test that loop raises error when neither condition nor over is set."""
    builder = PlanBuilderV2()

    with pytest.raises(
        PlanBuilderError, match="Exactly one of while_, do_while_, or over must be set"
    ):
        builder.loop()


def test_end_loop_without_loop_raises_error() -> None:
    """Test that end_loop raises error when called without a loop."""
    builder = PlanBuilderV2()

    with pytest.raises(PlanBuilderError, match="endloop must be called from a loop block"):
        builder.end_loop()


def test_end_loop_with_conditional_block_raises_error() -> None:
    """Test that end_loop raises error when called from conditional block."""
    builder = PlanBuilderV2()

    builder.if_(lambda: True)
    with pytest.raises(PlanBuilderError, match="endloop must be called from a loop block"):
        builder.end_loop()


def test_build_with_missing_end_loop_raises_error() -> None:
    """Test that build raises error when loop is not closed."""
    builder = PlanBuilderV2()

    builder.loop(while_=lambda: True)
    with pytest.raises(PlanBuilderError, match="All blocks must be closed"):
        builder.build()


def test_loop_method_chaining() -> None:
    """Test that loop methods return self for chaining."""
    builder = PlanBuilderV2()

    result = (
        builder.loop(while_=lambda: True)
        .llm_step(task="Inside loop")
        .end_loop()
        .llm_step(task="After loop")
    )

    assert result is builder
    plan = builder.build()
    assert len(plan.steps) == 4  # loop, llm, end_loop, final_llm


def test_complex_nested_structure() -> None:
    """Test complex nested structure with loops and conditionals."""
    builder = PlanBuilderV2()

    result = (
        builder.input(name="items", description="List of items")
        .function_step(function=lambda: [1, 2, 3], step_name="generate_items")
        .loop(over=StepOutput("generate_items"), step_name="outer_loop")
        .if_(lambda item: item > 1)
        .loop(while_=lambda: True, step_name="inner_loop")
        .llm_step(task="Process item in inner loop")
        .end_loop(step_name="inner_end")
        .else_()
        .llm_step(task="Process item in else")
        .endif()
        .end_loop(step_name="outer_end")
        .llm_step(task="Final step")
    )

    assert result is builder
    plan = builder.build()
    assert (
        len(plan.steps) == 11
    )  # func, loop, if, loop, llm, end_loop, else, llm, endif, end_loop, final_llm
    assert len(plan.plan_inputs) == 1

    # Verify the structure
    steps = plan.steps

    # Check function step
    assert steps[0].step_name == "generate_items"

    # Check outer loop
    assert steps[1].step_name == "outer_loop"
    outer_loop_step = steps[1]
    assert isinstance(outer_loop_step, LoopStep)
    assert outer_loop_step.loop_type == LoopType.FOR_EACH

    # Check if step
    assert steps[2].step_name == "step_2"

    # Check inner loop
    assert steps[3].step_name == "inner_loop"
    inner_loop_step = steps[3]
    assert isinstance(inner_loop_step, LoopStep)
    assert inner_loop_step.loop_type == LoopType.WHILE

    # Check inner loop content
    assert steps[4].step_name == "step_4"

    # Check inner end_loop
    assert steps[5].step_name == "inner_end"

    # Check else step
    assert steps[6].step_name == "step_6"

    # Check else content
    assert steps[7].step_name == "step_7"

    # Check endif
    assert steps[8].step_name == "step_8"

    # Check outer end_loop
    assert steps[9].step_name == "outer_end"

    # Check final step
    assert steps[10].step_name == "step_10"


def test_current_loop_block_property() -> None:
    """Test the _current_loop_block property."""
    builder = PlanBuilderV2()

    # Initially, no loop block should be current
    assert builder._current_loop_block is None

    # Start a loop
    builder.loop(while_=lambda: True)

    # Now there should be a current loop block
    current_block = builder._current_loop_block
    assert current_block is not None
    assert isinstance(current_block, LoopBlock)
    assert current_block.start_step_index == 0

    # End the loop
    builder.end_loop()

    # After ending the loop, no current loop block
    assert builder._current_loop_block is None

    # Test with nested loops
    builder.loop(while_=lambda: True)
    outer_block = builder._current_loop_block
    assert outer_block is not None
    assert outer_block.start_step_index > 0  # Should have some index

    builder.loop(while_=lambda: True)
    inner_block = builder._current_loop_block
    assert inner_block is not None
    assert (
        inner_block.start_step_index > outer_block.start_step_index
    )  # Inner should be after outer

    # End inner loop
    builder.end_loop()
    # Should be back to outer loop
    assert builder._current_loop_block is outer_block

    # End outer loop
    builder.end_loop()
    # No current loop block
    assert builder._current_loop_block is None


def test_end_loop_no_loop_block_error() -> None:
    """Test that end_loop raises PlanBuilderError when no loop block exists."""
    builder = PlanBuilderV2()

    # Try to end a loop without starting one
    with pytest.raises(
        PlanBuilderError, match="endloop must be called from a loop block. Please add a loop first."
    ):
        builder.end_loop()


def test_end_loop_non_loop_step_error() -> None:
    """Test that end_loop raises PlanBuilderError when start step is not a LoopStep."""
    builder = PlanBuilderV2()

    # Add a non-loop step first
    builder.llm_step(task="Some step")

    # Start a loop (this will create a loop block)
    builder.loop(while_=lambda: True)

    # Manually manipulate the block stack to point to a non-loop step
    # This simulates a corrupted state where the start_step_index points to wrong step type
    loop_block = builder._current_loop_block
    assert loop_block is not None

    # Change the start_step_index to point to the LLM step instead of the loop step
    loop_block.start_step_index = 0  # Points to the LLM step, not the loop step

    # Now try to end the loop - it should fail because step 0 is not a LoopStep
    with pytest.raises(
        PlanBuilderError, match="The step at the start of the loop is not a LoopStep"
    ):
        builder.end_loop()
