"""Test the PlanV2 class."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from portia.builder.invoke_tool_step import InvokeToolStep
from portia.builder.llm_step import LLMStep
from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import StepV2
from portia.plan import Plan, PlanContext, PlanInput, Step


class OutputSchema(BaseModel):
    """Output schema for testing."""

    result: str
    count: int


class MockStepV2(StepV2):
    """Mock step for testing."""

    def __init__(self, step_name: str = "mock_step") -> None:
        """Initialize mock step."""
        super().__init__(step_name=step_name)

    async def run(self, run_data: Any) -> str:  # noqa: ANN401, ARG002
        """Mock run method."""
        return "mock result"

    def to_step_data(self, plan: PlanV2) -> Step:  # noqa: ARG002
        """Mock to_step_data method."""
        return Step(
            task=f"Mock task for {self.step_name}",
            output=f"${self.step_name}_output",
            tool_id="mock_tool",
        )


# Test cases for PlanV2


def test_initialization_default_values() -> None:
    """Test PlanV2 initialization with default values."""
    plan = PlanV2(steps=[])

    assert hasattr(plan.id, "uuid")  # PlanUUID should have a uuid attribute
    assert plan.steps == []
    assert plan.plan_inputs == []
    assert plan.summarize is False
    assert plan.final_output_schema is None
    assert plan.label == "Run the plan built with the Plan Builder"


def test_initialization_custom_values() -> None:
    """Test PlanV2 initialization with custom values."""
    mock_step = MockStepV2("custom_step")
    plan_input = PlanInput(name="test_input", description="Test input description")

    plan = PlanV2(
        steps=[mock_step],
        plan_inputs=[plan_input],
        summarize=True,
        final_output_schema=OutputSchema,
        label="Custom Plan Label",
    )

    assert len(plan.steps) == 1
    assert plan.steps[0] is mock_step
    assert len(plan.plan_inputs) == 1
    assert plan.plan_inputs[0] is plan_input
    assert plan.summarize is True
    assert plan.final_output_schema is OutputSchema
    assert plan.label == "Custom Plan Label"


def test_to_legacy_plan_basic() -> None:
    """Test the to_legacy_plan() method with basic setup."""
    mock_step = MockStepV2("test_step")
    plan_input = PlanInput(name="input1", description="Test input")
    plan_context = PlanContext(query="Test query", tool_ids=["mock_tool"])

    plan = PlanV2(
        steps=[mock_step],
        plan_inputs=[plan_input],
        final_output_schema=OutputSchema,
    )

    legacy_plan = plan.to_legacy_plan(plan_context)

    assert isinstance(legacy_plan, Plan)
    assert legacy_plan.id == plan.id
    assert legacy_plan.plan_context is plan_context
    assert len(legacy_plan.steps) == 1
    assert legacy_plan.plan_inputs == [plan_input]
    assert legacy_plan.structured_output_schema is OutputSchema


def test_to_legacy_plan_multiple_steps() -> None:
    """Test the to_legacy_plan() method with multiple steps."""
    step1 = MockStepV2("step_1")
    step2 = MockStepV2("step_2")
    plan_context = PlanContext(query="Multi-step query", tool_ids=["mock_tool"])

    plan = PlanV2(steps=[step1, step2])
    legacy_plan = plan.to_legacy_plan(plan_context)

    assert len(legacy_plan.steps) == 2
    assert legacy_plan.steps[0].task == "Mock task for step_1"
    assert legacy_plan.steps[1].task == "Mock task for step_2"


def test_step_output_name_with_step_index() -> None:
    """Test step_output_name() method with step index."""
    step1 = MockStepV2("first_step")
    step2 = MockStepV2("second_step")
    plan = PlanV2(steps=[step1, step2])

    assert plan.step_output_name(0) == "$step_0_output"
    assert plan.step_output_name(1) == "$step_1_output"


def test_step_output_name_with_negative_index() -> None:
    """Test step_output_name() method with negative step index."""
    step1 = MockStepV2("first_step")
    step2 = MockStepV2("second_step")
    step3 = MockStepV2("third_step")
    plan = PlanV2(steps=[step1, step2, step3])

    assert plan.step_output_name(-1) == "$step_2_output"
    assert plan.step_output_name(-2) == "$step_1_output"


def test_step_output_name_with_step_name() -> None:
    """Test step_output_name() method with step name."""
    step1 = MockStepV2("custom_step_name")
    step2 = MockStepV2("another_step")
    plan = PlanV2(steps=[step1, step2])

    assert plan.step_output_name("custom_step_name") == "$step_0_output"
    assert plan.step_output_name("another_step") == "$step_1_output"


def test_step_output_name_with_step_instance() -> None:
    """Test step_output_name() method with StepV2 instance."""
    step1 = MockStepV2("instance_step")
    step2 = MockStepV2("another_instance")
    plan = PlanV2(steps=[step1, step2])

    assert plan.step_output_name(step1) == "$step_0_output"
    assert plan.step_output_name(step2) == "$step_1_output"


def test_step_output_name_invalid_step_index() -> None:
    """Test step_output_name() method with invalid step index."""
    plan = PlanV2(steps=[MockStepV2("test_step")])

    # Invalid indices don't raise ValueError, they just get passed through
    result = plan.step_output_name(999)  # Invalid index
    assert result == "$step_999_output"


def test_step_output_name_invalid_step_name() -> None:
    """Test step_output_name() method with invalid step name."""
    plan = PlanV2(steps=[MockStepV2("valid_step")])

    with patch("portia.builder.plan_v2.logger") as mock_logger:
        result = plan.step_output_name("nonexistent_step")

        # Should return a UUID-based fallback name
        assert result.startswith("$unknown_step_output_")
        mock_logger().warning.assert_called_once()


def test_step_output_name_step_not_in_plan() -> None:
    """Test step_output_name() method with step instance not in plan."""
    plan = PlanV2(steps=[MockStepV2("in_plan")])
    external_step = MockStepV2("not_in_plan")

    with patch("portia.builder.plan_v2.logger") as mock_logger:
        result = plan.step_output_name(external_step)

        # Should return a UUID-based fallback name
        assert result.startswith("$unknown_step_output_")
        mock_logger().warning.assert_called_once()


def test_idx_by_name_valid_names() -> None:
    """Test idx_by_name() method with valid step names."""
    step1 = MockStepV2("first")
    step2 = MockStepV2("second")
    step3 = MockStepV2("third")
    plan = PlanV2(steps=[step1, step2, step3])

    assert plan.idx_by_name("first") == 0
    assert plan.idx_by_name("second") == 1
    assert plan.idx_by_name("third") == 2


def test_idx_by_name_invalid_name() -> None:
    """Test idx_by_name() method with invalid step name."""
    plan = PlanV2(steps=[MockStepV2("existing_step")])

    with pytest.raises(ValueError, match="Step nonexistent not found in plan"):
        plan.idx_by_name("nonexistent")


def test_idx_by_name_empty_plan() -> None:
    """Test idx_by_name() method with empty plan."""
    plan = PlanV2(steps=[])

    with pytest.raises(ValueError, match="Step any_name not found in plan"):
        plan.idx_by_name("any_name")


def test_plan_with_real_step_types() -> None:
    """Test PlanV2 with actual step types from the codebase."""
    llm_step = LLMStep(
        task="Test LLM task",
        step_name="llm_step",
    )
    tool_step = InvokeToolStep(
        tool="test_tool",
        step_name="tool_step",
    )

    plan = PlanV2(steps=[llm_step, tool_step])

    # Test step output names
    assert plan.step_output_name(0) == "$step_0_output"
    assert plan.step_output_name(1) == "$step_1_output"
    assert plan.step_output_name("llm_step") == "$step_0_output"
    assert plan.step_output_name("tool_step") == "$step_1_output"

    # Test idx_by_name
    assert plan.idx_by_name("llm_step") == 0
    assert plan.idx_by_name("tool_step") == 1


def test_plan_with_no_steps() -> None:
    """Test PlanV2 behavior with no steps."""
    plan = PlanV2(steps=[])

    # idx_by_name should raise ValueError for any name
    with pytest.raises(ValueError, match="Step any_name not found in plan"):
        plan.idx_by_name("any_name")

    # step_output_name with invalid index should return default format
    result = plan.step_output_name(0)
    assert result == "$step_0_output"


def test_plan_id_generation() -> None:
    """Test that each PlanV2 instance gets a unique ID."""
    plan1 = PlanV2(steps=[])
    plan2 = PlanV2(steps=[])

    assert plan1.id != plan2.id
    assert hasattr(plan1.id, "uuid")
    assert hasattr(plan2.id, "uuid")


def test_plan_with_complex_configuration() -> None:
    """Test PlanV2 with a complex configuration."""
    steps: list[StepV2] = [
        MockStepV2("data_collection"),
        MockStepV2("data_processing"),
        MockStepV2("analysis"),
        MockStepV2("reporting"),
    ]

    inputs = [
        PlanInput(name="data_source", description="Source of the data"),
        PlanInput(name="analysis_type", description="Type of analysis to perform"),
    ]

    plan = PlanV2(
        steps=steps,
        plan_inputs=inputs,
        summarize=True,
        final_output_schema=OutputSchema,
        label="Complex Data Analysis Plan",
    )

    # Test all step names can be found
    for i, step in enumerate(steps):
        assert plan.idx_by_name(step.step_name) == i
        assert plan.step_output_name(i) == f"$step_{i}_output"
        assert plan.step_output_name(step.step_name) == f"$step_{i}_output"
        assert plan.step_output_name(step) == f"$step_{i}_output"

    # Test legacy plan conversion
    plan_context = PlanContext(
        query="Analyze complex data",
        tool_ids=["mock_tool"],
    )
    legacy_plan = plan.to_legacy_plan(plan_context)

    assert len(legacy_plan.steps) == 4
    assert len(legacy_plan.plan_inputs) == 2
    assert legacy_plan.structured_output_schema is OutputSchema


def test_validation_duplicate_step_names() -> None:
    """Test that duplicate step names raise a validation error."""
    steps: list[StepV2] = [
        MockStepV2("duplicate_name"),
        MockStepV2("unique_name"),
        MockStepV2("duplicate_name"),  # Duplicate
    ]

    with pytest.raises(ValueError):  # noqa: PT011
        PlanV2(steps=steps)


def test_validation_duplicate_plan_input_names() -> None:
    """Test that duplicate plan input names raise a validation error."""
    inputs = [
        PlanInput(name="duplicate_input", description="First input"),
        PlanInput(name="unique_input", description="Unique input"),
        PlanInput(name="duplicate_input", description="Second input with same name"),
    ]

    with pytest.raises(ValueError):  # noqa: PT011
        PlanV2(steps=[], plan_inputs=inputs)


def test_validation_multiple_duplicate_step_names() -> None:
    """Test validation with multiple different duplicate step names."""
    steps: list[StepV2] = [
        MockStepV2("dup1"),
        MockStepV2("dup2"),
        MockStepV2("unique"),
        MockStepV2("dup1"),  # Duplicate
        MockStepV2("dup2"),  # Duplicate
    ]

    with pytest.raises(ValueError):  # noqa: PT011
        PlanV2(steps=steps)


def test_validation_multiple_duplicate_input_names() -> None:
    """Test validation with multiple different duplicate input names."""
    inputs = [
        PlanInput(name="dup1", description="First"),
        PlanInput(name="dup2", description="Second"),
        PlanInput(name="unique", description="Unique"),
        PlanInput(name="dup1", description="Duplicate first"),
        PlanInput(name="dup2", description="Duplicate second"),
    ]

    with pytest.raises(ValueError, match="Duplicate plan input names found:"):
        PlanV2(steps=[], plan_inputs=inputs)


def test_validation_no_duplicates_passes() -> None:
    """Test that plans with no duplicates pass validation."""
    steps: list[StepV2] = [
        MockStepV2("step1"),
        MockStepV2("step2"),
        MockStepV2("step3"),
    ]
    inputs = [
        PlanInput(name="input1", description="First input"),
        PlanInput(name="input2", description="Second input"),
    ]

    # Should not raise any exception
    plan = PlanV2(steps=steps, plan_inputs=inputs)
    assert len(plan.steps) == 3
    assert len(plan.plan_inputs) == 2


def test_validation_empty_plan_passes() -> None:
    """Test that empty plans pass validation."""
    # Should not raise any exception
    plan = PlanV2(steps=[], plan_inputs=[])
    assert len(plan.steps) == 0
    assert len(plan.plan_inputs) == 0


def test_pretty_plan() -> None:
    """Test pretty print."""
    plan = PlanV2(
        steps=[MockStepV2("step1")],
        plan_inputs=[PlanInput(name="input1", description="First input")],
        final_output_schema=OutputSchema,
    )

    output = plan.pretty_print()
    assert isinstance(output, str)
