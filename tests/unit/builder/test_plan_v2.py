"""Test the PlanV2 class."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import InvokeToolStep, LLMStep, StepV2
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

    def describe(self) -> str:
        """Mock describe method."""
        return f"MockStep(step_name='{self.step_name}')"

    def to_legacy_step(self, plan: PlanV2) -> Step:  # noqa: ARG002
        """Mock to_legacy_step method."""
        return Step(
            task=f"Mock task for {self.step_name}",
            output=f"${self.step_name}_output",
            tool_id="mock_tool",
        )


class TestPlanV2:
    """Test cases for PlanV2."""

    def test_initialization_default_values(self) -> None:
        """Test PlanV2 initialization with default values."""
        plan = PlanV2(steps=[])

        assert hasattr(plan.id, "uuid")  # PlanUUID should have a uuid attribute
        assert plan.steps == []
        assert plan.plan_inputs == []
        assert plan.summarize is False
        assert plan.final_output_schema is None
        assert plan.label == "Run the plan built with the Plan Builder"

    def test_initialization_custom_values(self) -> None:
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

    def test_to_legacy_plan_basic(self) -> None:
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

    def test_to_legacy_plan_multiple_steps(self) -> None:
        """Test the to_legacy_plan() method with multiple steps."""
        step1 = MockStepV2("step_1")
        step2 = MockStepV2("step_2")
        plan_context = PlanContext(query="Multi-step query", tool_ids=["mock_tool"])

        plan = PlanV2(steps=[step1, step2])
        legacy_plan = plan.to_legacy_plan(plan_context)

        assert len(legacy_plan.steps) == 2
        assert legacy_plan.steps[0].task == "Mock task for step_1"
        assert legacy_plan.steps[1].task == "Mock task for step_2"

    def test_idx_by_name_valid_names(self) -> None:
        """Test idx_by_name() method with valid step names."""
        step1 = MockStepV2("first")
        step2 = MockStepV2("second")
        step3 = MockStepV2("third")
        plan = PlanV2(steps=[step1, step2, step3])

        assert plan.idx_by_name("first") == 0
        assert plan.idx_by_name("second") == 1
        assert plan.idx_by_name("third") == 2

    def test_idx_by_name_invalid_name(self) -> None:
        """Test idx_by_name() method with invalid step name."""
        plan = PlanV2(steps=[MockStepV2("existing_step")])

        with pytest.raises(ValueError, match="Step nonexistent not found in plan"):
            plan.idx_by_name("nonexistent")

    def test_idx_by_name_empty_plan(self) -> None:
        """Test idx_by_name() method with empty plan."""
        plan = PlanV2(steps=[])

        with pytest.raises(ValueError, match="Step any_name not found in plan"):
            plan.idx_by_name("any_name")

    def test_plan_with_real_step_types(self) -> None:
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

        assert plan.idx_by_name("llm_step") == 0
        assert plan.idx_by_name("tool_step") == 1

    def test_plan_with_no_steps(self) -> None:
        """Test PlanV2 behavior with no steps."""
        plan = PlanV2(steps=[])

        with pytest.raises(ValueError, match="Step any_name not found in plan"):
            plan.idx_by_name("any_name")

    def test_plan_id_generation(self) -> None:
        """Test that each PlanV2 instance gets a unique ID."""
        plan1 = PlanV2(steps=[])
        plan2 = PlanV2(steps=[])

        assert plan1.id != plan2.id
        assert hasattr(plan1.id, "uuid")
        assert hasattr(plan2.id, "uuid")

    def test_plan_with_complex_configuration(self) -> None:
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

        # Test legacy plan conversion
        plan_context = PlanContext(
            query="Analyze complex data",
            tool_ids=["mock_tool"],
        )
        legacy_plan = plan.to_legacy_plan(plan_context)

        assert len(legacy_plan.steps) == 4
        assert len(legacy_plan.plan_inputs) == 2
        assert legacy_plan.structured_output_schema is OutputSchema

    def test_validation_duplicate_step_names(self) -> None:
        """Test that duplicate step names raise a validation error."""
        steps: list[StepV2] = [
            MockStepV2("duplicate_name"),
            MockStepV2("unique_name"),
            MockStepV2("duplicate_name"),  # Duplicate
        ]

        with pytest.raises(ValueError):  # noqa: PT011
            PlanV2(steps=steps)

    def test_validation_duplicate_plan_input_names(self) -> None:
        """Test that duplicate plan input names raise a validation error."""
        inputs = [
            PlanInput(name="duplicate_input", description="First input"),
            PlanInput(name="unique_input", description="Unique input"),
            PlanInput(name="duplicate_input", description="Second input with same name"),
        ]

        with pytest.raises(ValueError):  # noqa: PT011
            PlanV2(steps=[], plan_inputs=inputs)

    def test_validation_multiple_duplicate_step_names(self) -> None:
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

    def test_validation_multiple_duplicate_input_names(self) -> None:
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

    def test_validation_no_duplicates_passes(self) -> None:
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

    def test_validation_empty_plan_passes(self) -> None:
        """Test that empty plans pass validation."""
        # Should not raise any exception
        plan = PlanV2(steps=[], plan_inputs=[])
        assert len(plan.steps) == 0
        assert len(plan.plan_inputs) == 0
