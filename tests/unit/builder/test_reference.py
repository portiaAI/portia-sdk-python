"""Test the reference module."""

from __future__ import annotations

from unittest.mock import Mock, patch

from pydantic import BaseModel

from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import AllStepOutputs, Input, StepOutput, default_step_name
from portia.builder.step_v2 import LLMStep, StepV2
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from portia.run_context import StepOutputValue

# Test cases for the default_step_name function


def test_default_step_name() -> None:
    """Test default_step_name."""
    assert default_step_name(0) == "step_0"
    assert default_step_name(1) == "step_1"
    assert default_step_name(5) == "step_5"
    assert default_step_name(42) == "step_42"
    assert default_step_name(999) == "step_999"
    assert default_step_name(123456) == "step_123456"


# Test cases for the StepOutput class


def test_step_output_initialization_with_int() -> None:
    """Test StepOutput initialization with integer step index."""
    step_output = StepOutput(5)
    assert step_output.step == 5


def test_step_output_initialization_with_string() -> None:
    """Test StepOutput initialization with string step name."""
    step_output = StepOutput("my_step")
    assert step_output.step == "my_step"


def test_step_output_with_path_initialization() -> None:
    """Test StepOutput initialization with path."""
    step_output = StepOutput("my_step", path="field.name")
    assert step_output.step == "my_step"
    assert step_output.path == "field.name"


def test_step_output_str_representation_int() -> None:
    """Test StepOutput string representation with integer step."""
    step_output = StepOutput(3)
    result = str(step_output)
    assert result == "{{ StepOutput(3) }}"


def test_step_output_str_representation_string() -> None:
    """Test StepOutput string representation with string step."""
    step_output = StepOutput("custom_step")
    result = str(step_output)
    assert result == "{{ StepOutput('custom_step') }}"


def test_step_output_with_path_str_representation() -> None:
    """Test StepOutput string representation with path."""
    step_output = StepOutput("my_step", path="field.name")
    result = str(step_output)
    assert result == "{{ StepOutput('my_step', path='field.name') }}"


def test_get_legacy_name_with_int_step() -> None:
    """Test get_legacy_name method with integer step."""
    step_output = StepOutput(2)

    # Create a mock plan that returns a specific output name
    mock_plan = Mock(spec=PlanV2)
    mock_plan.step_output_name.return_value = "$step_2_output"

    result = step_output.get_legacy_name(mock_plan)

    assert result == "$step_2_output"
    mock_plan.step_output_name.assert_called_once_with(2)


def test_get_legacy_name_with_string_step() -> None:
    """Test get_legacy_name method with string step."""
    step_output = StepOutput("named_step")

    # Create a mock plan that returns a specific output name
    mock_plan = Mock(spec=PlanV2)
    mock_plan.step_output_name.return_value = "$named_step_output"

    result = step_output.get_legacy_name(mock_plan)

    assert result == "$named_step_output"
    mock_plan.step_output_name.assert_called_once_with("named_step")


def test_get_value_with_int_step_success() -> None:
    """Test get_value method with integer step - successful case."""
    step_output = StepOutput(1)

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="test", description=""),
        StepOutputValue(step_num=1, step_name="step_1", value="test result", description=""),
        StepOutputValue(step_num=2, step_name="step_2", value="test", description=""),
    ]

    result = step_output.get_value(mock_run_data)

    assert result == "test result"


def test_get_value_with_string_step_success() -> None:
    """Test get_value method with string step - successful case."""
    step_output = StepOutput("my_step")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="test", description=""),
        StepOutputValue(step_num=1, step_name="step_1", value="test", description=""),
        StepOutputValue(step_num=2, step_name="my_step", value="test result", description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "test result"


def test_get_value_with_path_success() -> None:
    """Test get_value method with path - successful case."""

    class MockData:
        def __init__(self) -> None:
            self.field_a = MockDataFieldA()

    class MockDataFieldA:
        def __init__(self) -> None:
            self.field_b = "extracted_value"

    step_output = StepOutput(0, path="field_a.field_b")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=MockData(), description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "extracted_value"


def test_get_value_with_path_dict_access() -> None:
    """Test get_value method with path - successful dict access."""
    step_output = StepOutput("search", path="results.0.title")

    mock_data = {"results": [{"title": "First Result"}, {"title": "Second Result"}]}
    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="search", value=mock_data, description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "First Result"


def test_get_value_with_path_array_access() -> None:
    """Test get_value method with path - array indexing."""
    step_output = StepOutput(0, path="items.1.name")

    mock_data = {"items": [{"name": "first"}, {"name": "second"}, {"name": "third"}]}
    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=mock_data, description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "second"


def test_get_value_with_path_complex_key() -> None:
    """Test get_value method with path - complex key access with dot notation."""
    step_output = StepOutput(0, path="data.complex-key.value")

    mock_data = {"data": {"complex-key": {"value": "found_it"}}}
    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=mock_data, description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "found_it"


def test_get_value_with_path_mixed_notation() -> None:
    """Test get_value method with path - mixed object and array access."""

    class MockResult:
        def __init__(self) -> None:
            self.results = [{"title": "First"}, {"title": "Second"}]

    step_output = StepOutput(0, path="results.0.title")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=MockResult(), description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "First"


def test_get_value_with_int_step_index_error() -> None:
    """Test get_value method with integer step - IndexError case."""
    step_output = StepOutput(5)  # Index out of range

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0,
            step_name="step_0",
            value="test",
            description="",
        ),
        StepOutputValue(
            step_num=1,
            step_name="step_1",
            value="test",
            description="",
        ),
    ]

    result = step_output.get_value(mock_run_data)

    assert result is None


def test_get_value_with_string_step_value_error() -> None:
    """Test get_value method with string step - ValueError case."""
    step_output = StepOutput("nonexistent_step")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="test", description=""),
        StepOutputValue(step_num=1, step_name="step_1", value="test", description=""),
    ]

    result = step_output.get_value(mock_run_data)

    assert result is None


def test_get_value_with_path_attribute_error() -> None:
    """Test get_value method with path - returns None for missing attributes."""

    class MockData:
        def __init__(self) -> None:
            self.field_a = "simple_value"  # No field_b attribute

    step_output = StepOutput(0, path="field_a.field_b")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=MockData(), description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result is None


def test_get_value_with_path_key_error() -> None:
    """Test get_value method with path - returns None for missing keys."""
    step_output = StepOutput(0, path="nonexistent_key")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0, step_name="step_0", value={"existing_key": "value"}, description=""
        ),
    ]

    result = step_output.get_value(mock_run_data)
    assert result is None


def test_get_value_with_path_index_error() -> None:
    """Test get_value method with path - returns None for index out of range."""
    step_output = StepOutput(0, path="10")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=[1, 2, 3], description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result is None


def test_get_value_with_path_step_not_found() -> None:
    """Test get_value method with path when base step is not found."""
    step_output = StepOutput("nonexistent_step", path="field_name")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="test", description=""),
    ]

    with patch("portia.builder.reference.logger") as mock_logger:
        result = step_output.get_value(mock_run_data)

        assert result is None
        # Should log the step not found error, not the path error
        mock_logger().warning.assert_called_once_with(
            "Output value for step nonexistent_step not found"
        )


def test_get_value_with_path_pydantic_model() -> None:
    """Test get_value method with path - successful case with Pydantic model."""

    class UserProfile(BaseModel):
        name: str
        email: str

    class UserData(BaseModel):
        profile: UserProfile
        age: int

    step_output = StepOutput(0, path="profile.name")

    mock_data = UserData(profile=UserProfile(name="John Doe", email="john@example.com"), age=30)
    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value=mock_data, description=""),
    ]

    result = step_output.get_value(mock_run_data)
    assert result == "John Doe"


def test_get_description_with_int_step() -> None:
    """Test get_description method with integer step - successful case."""
    step_output = StepOutput(0)

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0, step_name="step_0", value="test", description="First step output"
        ),
    ]

    result = step_output.get_description(mock_run_data)

    assert result == "First step output"


def test_get_description_with_string_step() -> None:
    """Test get_description method with string step - successful case."""
    step_output = StepOutput("my_step")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0, step_name="step_0", value="test", description="First step output"
        ),
        StepOutputValue(
            step_num=1, step_name="my_step", value="test", description="Second step output"
        ),
    ]

    result = step_output.get_description(mock_run_data)
    assert result == "Second step output"


def test_get_description_with_invalid_step() -> None:
    """Test get_description method with invalid step - returns empty string."""
    step_output = StepOutput("nonexistent_step")

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(
            step_num=0, step_name="step_0", value="test", description="First step output"
        ),
        StepOutputValue(
            step_num=1, step_name="step_1", value="test", description="Second step output"
        ),
    ]

    result = step_output.get_description(mock_run_data)

    assert result == ""


# Test cases for the AllStepOutputs class


def test_all_step_outputs_str_representation() -> None:
    """Test AllStepOutputs string representation."""
    ref = AllStepOutputs()
    assert str(ref) == "{{ AllStepOutputs() }}"


def test_all_step_outputs_get_value() -> None:
    """Test AllStepOutputs get_value method."""
    ref = AllStepOutputs()

    mock_run_data = Mock()
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_0", value="foo", description=""),
        StepOutputValue(step_num=1, step_name="step_1", value=42, description=""),
    ]

    result = ref.get_value(mock_run_data)
    assert result == {"step_0": "foo", "step_1": 42}


def test_all_step_outputs_get_legacy_name() -> None:
    """Test AllStepOutputs get_legacy_name method."""
    ref = AllStepOutputs()

    # Create a mock plan with 2 steps
    mock_plan = Mock(spec=PlanV2)
    mock_plan.steps = [Mock(), Mock()]  # 2 mock steps
    mock_plan.step_output_name.side_effect = lambda i: f"$step_{i}_output"

    result = ref.get_legacy_name(mock_plan)
    assert result == "$step_0_output,$step_1_output"


def test_all_step_outputs_get_description() -> None:
    """Test AllStepOutputs get_description method."""
    ref = AllStepOutputs()
    mock_run_data = Mock()

    result = ref.get_description(mock_run_data)
    assert result == "All previous step outputs"


# Test cases for the Input class


def test_input_initialization() -> None:
    """Test Input initialization."""
    input_ref = Input("user_name")
    assert input_ref.name == "user_name"


def test_input_str_representation() -> None:
    """Test Input string representation."""
    input_ref = Input("api_key")
    result = str(input_ref)
    assert result == "{{ Input('api_key') }}"


def test_input_with_path_initialization() -> None:
    """Test Input initialization with path."""
    input_ref = Input("user_data", path="profile.name")
    assert input_ref.name == "user_data"
    assert input_ref.path == "profile.name"


def test_input_with_path_str_representation() -> None:
    """Test Input string representation with path."""
    input_ref = Input("user_data", path="profile.name")
    result = str(input_ref)
    assert result == "{{ Input('user_data', path='profile.name') }}"


def test_get_legacy_name() -> None:
    """Test get_legacy_name method."""
    input_ref = Input("my_input")

    result = input_ref.get_legacy_name(Mock(spec=PlanV2))

    assert result == "my_input"


def test_get_value_success() -> None:
    """Test get_value method - successful case."""
    input_ref = Input("user_name")

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_name", description="The user's name")

    # Create mock output value
    test_output = LocalDataValue(value="John Doe", summary="User name")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_name": test_output}

    result = input_ref.get_value(mock_run_data)

    assert result == "John Doe"


def test_get_value_success_no_description() -> None:
    """Test get_value method - successful case with no description."""
    input_ref = Input("api_key")

    # Create mock plan input without description
    mock_plan_input = PlanInput(name="api_key", description=None)

    # Create mock output value
    test_output = LocalDataValue(value="secret-key-123", summary="API key")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"api_key": test_output}

    result = input_ref.get_value(mock_run_data)

    assert result == "secret-key-123"


def test_get_value_with_reference() -> None:
    """Test get_value method - successful case with no description."""
    input_ref = Input("api_key")

    # Create mock plan input with a reference to a step output.
    # This can haopen in nested sub-plans
    mock_plan_input = PlanInput(name="api_key", value=StepOutput(0))

    # Create mock output value
    test_output = LocalDataValue(value=StepOutput(0))

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"api_key": test_output}
    mock_run_data.step_output_values = [
        StepOutputValue(step_num=0, step_name="step_output_0", value="test", description=""),
    ]

    result = input_ref.get_value(mock_run_data)

    assert result == "test"


def test_get_value_input_not_found_in_plan() -> None:
    """Test get_value method - input not found in plan."""
    input_ref = Input("missing_input")

    # Create mock run data with different input
    mock_plan_input = PlanInput(name="other_input", description="Other input")
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]

    result = input_ref.get_value(mock_run_data)

    assert result is None


def test_get_value_value_not_found_in_run_inputs() -> None:
    """Test get_value method - value not found in plan run inputs."""
    input_ref = Input("user_name")

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_name", description="The user's name")

    # Create mock run data without the value in plan_run_inputs
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {}  # Empty inputs

    result = input_ref.get_value(mock_run_data)

    assert result is None


def test_get_value_value_is_none_in_run_inputs() -> None:
    """Test get_value method - value is None in plan run inputs."""
    input_ref = Input("optional_input")

    # Create mock plan input
    mock_plan_input = PlanInput(name="optional_input", description="Optional input")

    # Create mock run data with None value
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"optional_input": None}

    with patch("portia.builder.reference.logger") as mock_logger:
        result = input_ref.get_value(mock_run_data)

        assert result is None
        mock_logger().warning.assert_called_once_with("Value not found for input optional_input")


def test_input_get_value_with_path_success() -> None:
    """Test Input get_value method with path - successful case."""

    class MockUserData:
        def __init__(self) -> None:
            self.profile = MockProfile()

    class MockProfile:
        def __init__(self) -> None:
            self.name = "John Doe"

    input_ref = Input("user_data", path="profile.name")

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_data", description="User data")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_data": LocalDataValue(value=MockUserData())}

    result = input_ref.get_value(mock_run_data)
    assert result == "John Doe"


def test_input_get_value_with_path_dict_access() -> None:
    """Test Input get_value method with path - successful dict access."""
    input_ref = Input("user_data", path="profile.email")

    mock_data = {"profile": {"email": "john@example.com", "name": "John"}}

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_data", description="User profile data")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_data": LocalDataValue(value=mock_data)}

    result = input_ref.get_value(mock_run_data)
    assert result == "john@example.com"


def test_input_get_value_with_path_array_access() -> None:
    """Test Input get_value method with path - array indexing."""
    input_ref = Input("user_data", path="preferences.0.category")

    mock_data = {"preferences": [{"category": "tech"}, {"category": "sports"}]}

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_data", description="User preferences")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_data": LocalDataValue(value=mock_data)}

    result = input_ref.get_value(mock_run_data)
    assert result == "tech"


def test_input_get_value_with_path_missing_attribute() -> None:
    """Test Input get_value method with path - returns None for missing attributes."""

    class MockUserData:
        def __init__(self) -> None:
            self.profile = "simple_value"  # No name attribute

    input_ref = Input("user_data", path="profile.name")

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_data", description="User data")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_data": LocalDataValue(value=MockUserData())}

    result = input_ref.get_value(mock_run_data)
    assert result is None


def test_input_get_value_with_path_missing_key() -> None:
    """Test Input get_value method with path - returns None for missing keys."""
    input_ref = Input("user_data", path="nonexistent_key")

    mock_data = {"existing_key": "value"}

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_data", description="User data")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_data": LocalDataValue(value=mock_data)}

    result = input_ref.get_value(mock_run_data)
    assert result is None


def test_input_get_value_with_path_index_out_of_range() -> None:
    """Test Input get_value method with path - returns None for index out of range."""
    input_ref = Input("user_data", path="items.5")

    mock_data = {"items": [1, 2, 3]}

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_data", description="User data")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_data": LocalDataValue(value=mock_data)}

    result = input_ref.get_value(mock_run_data)
    assert result is None


def test_input_get_value_with_path_pydantic_model() -> None:
    """Test Input get_value method with path - successful case with Pydantic model."""

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class UserProfile(BaseModel):
        name: str
        email: str
        address: Address

    input_ref = Input("user_profile", path="address.city")

    mock_data = UserProfile(
        name="Jane Smith",
        email="jane@example.com",
        address=Address(street="123 Main St", city="New York", country="USA"),
    )

    # Create mock plan input
    mock_plan_input = PlanInput(name="user_profile", description="User profile data")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]
    mock_run_data.plan_run.plan_run_inputs = {"user_profile": LocalDataValue(value=mock_data)}

    result = input_ref.get_value(mock_run_data)
    assert result == "New York"


def test_input_get_description_mainline() -> None:
    """Test Input get_description method - mainline case with description."""
    input_ref = Input("user_name")

    # Create mock plan input with description
    mock_plan_input = PlanInput(name="user_name", description="The user's name")

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]

    result = input_ref.get_description(mock_run_data)
    assert result == "The user's name"


def test_input_get_description_reference_case() -> None:
    """Test Input get_description method - reference case where value is StepOutput."""
    input_ref = Input("api_key")

    # Create mock plan input with a reference to a step output as value
    mock_step_output = Mock(spec=StepOutput)
    mock_step_output.get_description.return_value = "Output from step 0"
    mock_plan_input = PlanInput(name="api_key", value=mock_step_output)

    # Create mock run data
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]

    result = input_ref.get_description(mock_run_data)

    assert result == "Output from step 0"
    mock_step_output.get_description.assert_called_once_with(mock_run_data)


def test_input_get_description_not_found() -> None:
    """Test Input get_description method - input not found case."""
    input_ref = Input("missing_input")

    # Create mock run data with different input
    mock_plan_input = PlanInput(name="other_input", description="Other input")
    mock_run_data = Mock()
    mock_run_data.plan.plan_inputs = [mock_plan_input]

    result = input_ref.get_description(mock_run_data)
    assert result == ""


# Integration tests for reference classes


def test_step_output_and_input_with_real_plan() -> None:
    """Test StepOutput and Input with a real PlanV2 instance."""
    # Create a real plan with steps and inputs
    step = LLMStep(task="Test task", step_name="test_step")
    plan_input = PlanInput(name="test_input", description="Test input")
    plan = PlanV2(steps=[step], plan_inputs=[plan_input])

    # Test StepOutput
    step_output = StepOutput(0)
    legacy_name = step_output.get_legacy_name(plan)
    assert legacy_name == "$step_0_output"

    step_output_by_name = StepOutput("test_step")
    legacy_name_by_name = step_output_by_name.get_legacy_name(plan)
    assert legacy_name_by_name == "$step_0_output"

    # Test Input
    input_ref = Input("test_input")
    legacy_input_name = input_ref.get_legacy_name(plan)
    assert legacy_input_name == "test_input"


def test_multiple_inputs_and_outputs() -> None:
    """Test with multiple inputs and step outputs."""
    # Create plan with multiple steps and inputs
    steps: list[StepV2] = [
        LLMStep(task="First task", step_name="first_step"),
        LLMStep(task="Second task", step_name="second_step"),
        LLMStep(task="Third task", step_name="third_step"),
    ]
    inputs = [
        PlanInput(name="input1", description="First input"),
        PlanInput(name="input2", description="Second input"),
    ]
    plan = PlanV2(steps=steps, plan_inputs=inputs)

    # Test various StepOutput references
    assert StepOutput(0).get_legacy_name(plan) == "$step_0_output"
    assert StepOutput(1).get_legacy_name(plan) == "$step_1_output"
    assert StepOutput(2).get_legacy_name(plan) == "$step_2_output"

    assert StepOutput("first_step").get_legacy_name(plan) == "$step_0_output"
    assert StepOutput("second_step").get_legacy_name(plan) == "$step_1_output"
    assert StepOutput("third_step").get_legacy_name(plan) == "$step_2_output"

    # Test Input references
    assert Input("input1").get_legacy_name(plan) == "input1"
    assert Input("input2").get_legacy_name(plan) == "input2"
