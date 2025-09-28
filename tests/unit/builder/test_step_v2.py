"""Test the step_v2 module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

from portia.builder.llm_step import LLMStep
from portia.builder.reference import Input, StepOutput
from portia.builder.step_v2 import StepV2
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput, Variable
from portia.plan import Step as PlanStep
from portia.run_context import StepOutputValue

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2


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
    mock_run_data.plan.steps = []

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
    mock_run_data.plan.steps = []
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


def test_string_templating_with_path_comprehensive() -> None:
    """Test comprehensive string templating with path syntax for various scenarios."""
    step = LLMStep(task="Test", step_name="test_step")

    # Set up mock run data with complex nested structures
    mock_run_data = Mock()
    mock_run_data.plan = Mock()
    mock_run_data.plan.plan_inputs = [
        PlanInput(name="user_profile", description="User profile data"),
        PlanInput(name="preferences", description="User preferences"),
    ]
    mock_run_data.plan_run = Mock()
    mock_run_data.plan_run.plan_run_inputs = {
        "user_profile": LocalDataValue(
            value={
                "profile": {"name": "Alice", "email": "alice@example.com"},
                "settings": {"theme": "dark", "notifications": True},
            }
        ),
        "preferences": LocalDataValue(
            value=[{"category": "tech", "priority": 1}, {"category": "sports", "priority": 2}]
        ),
    }
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0,
            step_name="search",
            value={"results": [{"title": "AI Research", "score": 0.95}]},
            description="Search results",
        ),
        StepOutputValue(
            step_num=1,
            step_name="analysis",
            value={"data": {"items": {"complex-key": {"value": "nested_data"}}}},
            description="Analysis results",
        ),
    ]
    mock_run_data.plan.steps = []

    # Test 1: StepOutput with simple path
    result = step._template_references(
        "Found: {{ StepOutput('search', path='results.0.title') }}", mock_run_data
    )
    assert result == "Found: AI Research"

    # Test 2: StepOutput with complex path (special characters in key)
    result = step._template_references(
        "Data: {{ StepOutput('analysis', path='data.items.complex-key.value') }}", mock_run_data
    )
    assert result == "Data: nested_data"

    # Test 3: Input with path accessing nested object
    result = step._template_references(
        "User: {{ Input('user_profile', path='profile.name') }}", mock_run_data
    )
    assert result == "User: Alice"

    # Test 4: Input with path accessing array element
    result = step._template_references(
        "Top preference: {{ Input('preferences', path='0.category') }}", mock_run_data
    )
    assert result == "Top preference: tech"

    # Test 5: Mixed StepOutput and Input references with paths
    result = step._template_references(
        "{{ Input('user_profile', path='profile.name') }} searched for "
        "{{ StepOutput('search', path='results.0.title') }} with score "
        "{{ StepOutput('search', path='results.0.score') }}",
        mock_run_data,
    )
    assert result == "Alice searched for AI Research with score 0.95"

    # Test 6: Mixed quote styles (single and double quotes)
    result = step._template_references(
        'Email: {{ Input("user_profile", path="profile.email") }}', mock_run_data
    )
    assert result == "Email: alice@example.com"

    # Test 7: References without paths mixed with path references
    mock_run_data.step_output_values.append(
        StepOutputValue(
            step_num=2,
            step_name="simple",
            value="simple_result",
            description="Simple result",
        )
    )
    result = step._template_references(
        "Simple: {{ StepOutput('simple') }} vs Complex: "
        "{{ StepOutput('analysis', path='data.items.complex-key.value') }}",
        mock_run_data,
    )
    assert result == "Simple: simple_result vs Complex: nested_data"

    # Test 8: Numeric step references with paths
    result = step._template_references(
        "Result from step 0: {{ StepOutput(0, path='results.0.title') }}", mock_run_data
    )
    assert result == "Result from step 0: AI Research"


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
