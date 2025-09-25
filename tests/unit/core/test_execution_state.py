"""Tests for execution state classes."""

from unittest.mock import Mock

import pytest

from portia.builder.plan_v2 import PlanV2
from portia.core.execution_state import PlanRunSession, StepOutputValue
from portia.end_user import EndUser
from portia.plan import Plan
from portia.plan_run import PlanRun


class TestStepOutputValue:
    """Test cases for StepOutputValue dataclass."""

    def test_step_output_value_creation(self):
        """Test that StepOutputValue can be created with all required fields."""
        output_value = StepOutputValue(
            value="test_value",
            description="A test value",
            step_name="test_step",
            step_num=1,
        )

        assert output_value.value == "test_value"
        assert output_value.description == "A test value"
        assert output_value.step_name == "test_step"
        assert output_value.step_num == 1

    def test_step_output_value_with_complex_value(self):
        """Test StepOutputValue with complex data types."""
        complex_value = {"nested": {"data": [1, 2, 3]}}
        output_value = StepOutputValue(
            value=complex_value,
            description="Complex nested data",
            step_name="complex_step",
            step_num=2,
        )

        assert output_value.value == complex_value
        assert output_value.value["nested"]["data"] == [1, 2, 3]


class TestPlanRunSession:
    """Test cases for PlanRunSession dataclass."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for PlanRunSession."""
        return {
            "plan": Mock(spec=PlanV2),
            "legacy_plan": Mock(spec=Plan),
            "plan_run": Mock(spec=PlanRun),
            "end_user": Mock(spec=EndUser),
            "step_output_values": [],
        }

    def test_plan_run_session_creation(self, mock_dependencies):
        """Test that PlanRunSession can be created with all dependencies."""
        session = PlanRunSession(**mock_dependencies)

        assert session.plan is mock_dependencies["plan"]
        assert session.legacy_plan is mock_dependencies["legacy_plan"]
        assert session.plan_run is mock_dependencies["plan_run"]
        assert session.end_user is mock_dependencies["end_user"]
        assert session.step_output_values == []

    def test_add_step_output(self, mock_dependencies):
        """Test adding step outputs to the session."""
        session = PlanRunSession(**mock_dependencies)

        session.add_step_output(
            value="output1", description="First output", step_name="step1", step_num=0
        )
        session.add_step_output(
            value="output2", description="Second output", step_name="step2", step_num=1
        )

        assert len(session.step_output_values) == 2
        assert session.step_output_values[0].value == "output1"
        assert session.step_output_values[0].step_name == "step1"
        assert session.step_output_values[1].value == "output2"
        assert session.step_output_values[1].step_name == "step2"

    def test_get_step_outputs_by_name(self, mock_dependencies):
        """Test retrieving step outputs by step name."""
        session = PlanRunSession(**mock_dependencies)

        # Add multiple outputs with same step name
        session.add_step_output(
            value="output1", description="First output", step_name="step1", step_num=0
        )
        session.add_step_output(
            value="output2", description="Second output", step_name="step1", step_num=2
        )
        session.add_step_output(
            value="output3", description="Third output", step_name="step2", step_num=1
        )

        step1_outputs = session.get_step_outputs_by_name("step1")
        assert len(step1_outputs) == 2
        assert step1_outputs[0].value == "output1"
        assert step1_outputs[1].value == "output2"

        step2_outputs = session.get_step_outputs_by_name("step2")
        assert len(step2_outputs) == 1
        assert step2_outputs[0].value == "output3"

        nonexistent_outputs = session.get_step_outputs_by_name("nonexistent")
        assert len(nonexistent_outputs) == 0

    def test_get_step_output_by_num(self, mock_dependencies):
        """Test retrieving step output by step number."""
        session = PlanRunSession(**mock_dependencies)

        session.add_step_output(
            value="output1", description="First output", step_name="step1", step_num=0
        )
        session.add_step_output(
            value="output2", description="Second output", step_name="step2", step_num=1
        )

        output_0 = session.get_step_output_by_num(0)
        assert output_0 is not None
        assert output_0.value == "output1"
        assert output_0.step_name == "step1"

        output_1 = session.get_step_output_by_num(1)
        assert output_1 is not None
        assert output_1.value == "output2"
        assert output_1.step_name == "step2"

        nonexistent_output = session.get_step_output_by_num(999)
        assert nonexistent_output is None

    def test_plan_run_session_mutability(self, mock_dependencies):
        """Test that PlanRunSession is mutable for step outputs."""
        session = PlanRunSession(**mock_dependencies)

        # Should be able to modify step_output_values list
        initial_length = len(session.step_output_values)
        session.step_output_values.append(
            StepOutputValue(
                value="test", description="test", step_name="test", step_num=0
            )
        )
        assert len(session.step_output_values) == initial_length + 1

    def test_step_output_value_types(self, mock_dependencies):
        """Test that various data types can be stored as step output values."""
        session = PlanRunSession(**mock_dependencies)

        # Test different data types
        test_cases = [
            ("string", "String value"),
            (42, "Integer value"),
            (3.14, "Float value"),
            ([1, 2, 3], "List value"),
            ({"key": "value"}, "Dict value"),
            (None, "None value"),
        ]

        for i, (value, description) in enumerate(test_cases):
            session.add_step_output(
                value=value,
                description=description,
                step_name=f"step_{i}",
                step_num=i,
            )

        # Verify all types were stored correctly
        assert len(session.step_output_values) == len(test_cases)
        for i, (expected_value, _) in enumerate(test_cases):
            output = session.get_step_output_by_num(i)
            assert output is not None
            assert output.value == expected_value