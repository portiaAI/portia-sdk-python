"""Test the reference module."""

from __future__ import annotations

from unittest.mock import Mock, patch

from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import Input, ReferenceValue, StepOutput, default_step_name
from portia.builder.step_v2 import LLMStep, StepV2
from portia.execution_agents.output import LocalDataValue
from portia.plan import PlanInput
from tests.unit.builder.test_plan_v2 import MockStepV2


class TestDefaultStepName:
    """Test cases for the default_step_name function."""

    def test_default_step_name(self) -> None:
        """Test default_step_name."""
        assert default_step_name(0) == "step_0"
        assert default_step_name(1) == "step_1"
        assert default_step_name(5) == "step_5"
        assert default_step_name(42) == "step_42"
        assert default_step_name(999) == "step_999"
        assert default_step_name(123456) == "step_123456"


class TestStepOutput:
    """Test cases for the StepOutput class."""

    def test_step_output_initialization_with_int(self) -> None:
        """Test StepOutput initialization with integer step index."""
        step_output = StepOutput(5)
        assert step_output.step == 5

    def test_step_output_initialization_with_string(self) -> None:
        """Test StepOutput initialization with string step name."""
        step_output = StepOutput("my_step")
        assert step_output.step == "my_step"

    def test_step_output_str_representation_int(self) -> None:
        """Test StepOutput string representation with integer step."""
        step_output = StepOutput(3)
        result = str(step_output)
        assert result == "{{ StepOutput(3) }}"

    def test_step_output_str_representation_string(self) -> None:
        """Test StepOutput string representation with string step."""
        step_output = StepOutput("custom_step")
        result = str(step_output)
        assert result == "{{ StepOutput(custom_step) }}"

    def test_get_legacy_name_with_int_step(self) -> None:
        """Test get_legacy_name method with integer step."""
        step_output = StepOutput(2)

        # Create a mock plan that returns a specific output name
        mock_plan = Mock(spec=PlanV2)
        mock_plan.step_output_name.return_value = "$step_2_output"

        result = step_output.get_legacy_name(mock_plan)

        assert result == "$step_2_output"
        mock_plan.step_output_name.assert_called_once_with(2)

    def test_get_legacy_name_with_string_step(self) -> None:
        """Test get_legacy_name method with string step."""
        step_output = StepOutput("named_step")

        # Create a mock plan that returns a specific output name
        mock_plan = Mock(spec=PlanV2)
        mock_plan.step_output_name.return_value = "$named_step_output"

        result = step_output.get_legacy_name(mock_plan)

        assert result == "$named_step_output"
        mock_plan.step_output_name.assert_called_once_with("named_step")

    def test_step_output_name_with_step_index(self) -> None:
        """Test step_output_name() method with step index."""
        assert StepOutput.name_from_step(0, Mock(spec=PlanV2)) == "$step_0_output"
        assert StepOutput.name_from_step(1, Mock(spec=PlanV2)) == "$step_1_output"

        # Invalid step still returns a valid string
        assert StepOutput.name_from_step(999, Mock(spec=PlanV2)) == "$step_999_output"

    def test_step_output_name_with_step_instance(self) -> None:
        """Test step_output_name() method with StepV2 instance."""
        step1 = MockStepV2("instance_step")
        step2 = MockStepV2("another_instance")
        plan = PlanV2(steps=[step1, step2])

        assert StepOutput.name_from_step(step1, plan) == "$step_0_output"
        assert StepOutput.name_from_step(step2, plan) == "$step_1_output"

        # Invalid step still returns a valid string
        assert StepOutput.name_from_step(MockStepV2("nonexistent"), plan).startswith(
            "$unknown_step_output_"
        )

    def test_step_output_name_with_step_name(self) -> None:
        """Test step_output_name() method with step name."""
        step1 = MockStepV2("custom_step_name")
        step2 = MockStepV2("another_step")
        plan = PlanV2(steps=[step1, step2])

        assert StepOutput.name_from_step("custom_step_name", plan) == "$step_0_output"
        assert StepOutput.name_from_step("another_step", plan) == "$step_1_output"

        # Invalid step still returns a valid string
        result = StepOutput.name_from_step("nonexistent_step", plan)
        assert result.startswith("$unknown_step_output_")

    def test_get_value_with_int_step_success(self) -> None:
        """Test get_value method with integer step - successful case."""
        step_output = StepOutput(1)

        # Create mock run data
        test_output = LocalDataValue(value="test result", summary="Test output")
        mock_reference_value = ReferenceValue(value=test_output, description="Test output")

        mock_run_data = Mock()
        mock_run_data.step_output_values = [None, mock_reference_value, None]  # Index 1 has value

        result = step_output.get_value(mock_run_data)

        assert result is mock_reference_value

    def test_get_value_with_string_step_success(self) -> None:
        """Test get_value method with string step - successful case."""
        step_output = StepOutput("my_step")

        # Create mock run data
        test_output = LocalDataValue(value="test result", summary="Test output")
        mock_reference_value = ReferenceValue(value=test_output, description="Test output")

        mock_run_data = Mock()
        mock_run_data.plan.idx_by_name.return_value = 2  # Step is at index 2
        mock_run_data.step_output_values = [None, None, mock_reference_value]  # Index 2 has value

        result = step_output.get_value(mock_run_data)

        assert result is mock_reference_value
        mock_run_data.plan.idx_by_name.assert_called_once_with("my_step")

    def test_get_value_with_int_step_index_error(self) -> None:
        """Test get_value method with integer step - IndexError case."""
        step_output = StepOutput(5)  # Index out of range

        mock_run_data = Mock()
        mock_run_data.step_output_values = [None, None]  # Only 2 elements, index 5 doesn't exist

        result = step_output.get_value(mock_run_data)

        assert result is None

    def test_get_value_with_string_step_value_error(self) -> None:
        """Test get_value method with string step - ValueError case."""
        step_output = StepOutput("nonexistent_step")

        mock_run_data = Mock()
        mock_run_data.plan.idx_by_name.side_effect = ValueError("Step not found")

        result = step_output.get_value(mock_run_data)

        assert result is None


class TestInput:
    """Test cases for the Input class."""

    def test_input_initialization(self) -> None:
        """Test Input initialization."""
        input_ref = Input("user_name")
        assert input_ref.name == "user_name"

    def test_input_str_representation(self) -> None:
        """Test Input string representation."""
        input_ref = Input("api_key")
        result = str(input_ref)
        assert result == "{{ Input(api_key) }}"

    def test_get_legacy_name(self) -> None:
        """Test get_legacy_name method."""
        input_ref = Input("my_input")

        result = input_ref.get_legacy_name(Mock(spec=PlanV2))

        assert result == "my_input"

    def test_get_value_success(self) -> None:
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

        assert isinstance(result, ReferenceValue)
        assert result.value is test_output
        assert result.description == "The user's name"

    def test_get_value_success_no_description(self) -> None:
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

        assert isinstance(result, ReferenceValue)
        assert result.value is test_output
        assert result.description == "Input to plan"  # Default description

    def test_get_value_input_not_found_in_plan(self) -> None:
        """Test get_value method - input not found in plan."""
        input_ref = Input("missing_input")

        # Create mock run data with different input
        mock_plan_input = PlanInput(name="other_input", description="Other input")
        mock_run_data = Mock()
        mock_run_data.plan.plan_inputs = [mock_plan_input]

        result = input_ref.get_value(mock_run_data)

        assert result is None

    def test_get_value_value_not_found_in_run_inputs(self) -> None:
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

    def test_get_value_value_is_none_in_run_inputs(self) -> None:
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
            mock_logger().warning.assert_called_once_with(
                "Value not found for input optional_input"
            )


class TestReferenceValue:
    """Test cases for the ReferenceValue class."""

    def test_reference_value_initialization(self) -> None:
        """Test ReferenceValue initialization."""
        test_output = LocalDataValue(value="test data", summary="Test summary")
        ref_value = ReferenceValue(value=test_output, description="Test description")

        assert ref_value.value is test_output
        assert ref_value.description == "Test description"

    def test_reference_value_default_description(self) -> None:
        """Test ReferenceValue with default description."""
        test_output = LocalDataValue(value="test data", summary="Test summary")
        ref_value = ReferenceValue(value=test_output)

        assert ref_value.value is test_output
        assert ref_value.description == ""


class TestIntegration:
    """Integration tests for reference classes."""

    def test_step_output_and_input_with_real_plan(self) -> None:
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

    def test_multiple_inputs_and_outputs(self) -> None:
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
