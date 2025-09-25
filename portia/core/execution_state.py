"""Lightweight dataclasses for execution state management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.end_user import EndUser
    from portia.plan import Plan
    from portia.plan_run import PlanRun


@dataclass
class StepOutputValue:
    """Value that can be referenced by name.

    This is a lightweight replacement for heterogeneous tuples used to track
    step outputs during plan execution.

    Attributes:
        value: The referenced value.
        description: Description of the referenced value.
        step_name: The name of the referenced value.
        step_num: The step number of the referenced value.
    """

    value: Any
    description: str
    step_name: str
    step_num: int


@dataclass
class PlanRunSession:
    """Execution state for a plan run session.

    This lightweight dataclass replaces heterogeneous tuples that are currently
    passed between methods during plan execution. It provides a structured way
    to manage execution state while maintaining type safety.

    Attributes:
        plan: The modern PlanV2 being executed.
        legacy_plan: The legacy Plan representation for backward compatibility.
        plan_run: The current plan run instance.
        end_user: The end user executing the plan.
        step_output_values: Outputs set by executed steps.
    """

    plan: PlanV2
    legacy_plan: Plan
    plan_run: PlanRun
    end_user: EndUser
    step_output_values: list[StepOutputValue]

    def add_step_output(
        self, value: Any, description: str, step_name: str, step_num: int
    ) -> None:
        """Add a step output value to the session.

        Args:
            value: The output value from the step.
            description: Description of the output.
            step_name: Name of the step that produced the output.
            step_num: Step number that produced the output.
        """
        self.step_output_values.append(
            StepOutputValue(
                value=value,
                description=description,
                step_name=step_name,
                step_num=step_num,
            )
        )

    def get_step_outputs_by_name(self, step_name: str) -> list[StepOutputValue]:
        """Get all step outputs with a specific name.

        Args:
            step_name: The name of the step to get outputs for.

        Returns:
            List of step output values with the given name.
        """
        return [output for output in self.step_output_values if output.step_name == step_name]

    def get_step_output_by_num(self, step_num: int) -> StepOutputValue | None:
        """Get step output by step number.

        Args:
            step_num: The step number to get output for.

        Returns:
            The step output value if found, None otherwise.
        """
        for output in self.step_output_values:
            if output.step_num == step_num:
                return output
        return None