"""References to values in a plan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field

from portia.execution_agents.output import Output
from portia.logger import logger

if TYPE_CHECKING:
    from portia.builder.portia_plan import PortiaPlan
    from portia.portia import RunData


def default_step_name(step_index: int) -> str:
    """Return the default name for the step."""
    return f"step_{step_index}"


class Reference(ABC):
    """A reference to a value."""

    @abstractmethod
    def get_legacy_name(self, plan: PortiaPlan) -> str:
        """Get the name of the reference to use with legacy Portia plans."""
        raise NotImplementedError

    @abstractmethod
    def get_value(self, run_data: RunData) -> ReferenceValue | None:
        """Get the value of the reference."""
        raise NotImplementedError


class StepOutput(Reference):
    """A reference to the output of a step."""

    step: str | int = Field(
        description="The step to reference the output of. If a string is provided, this will be"
        "used to find the step by name. If an integer is provided, this will be used to find the"
        "step by index (steps are 0-indexed)."
    )

    def __init__(self, step: str | int) -> None:
        """Initialize the step output."""
        self.step = step

    @override
    def get_legacy_name(self, plan: PortiaPlan) -> str:
        """Get the name of the reference to use with legacy Portia plans."""
        return plan.step_output_name(self.step)

    @override
    def get_value(self, run_data: RunData) -> ReferenceValue | None:
        """Get the value of the step output."""
        try:
            if isinstance(self.step, int):
                return run_data.step_output_values[self.step]
            step_index = run_data.plan.idx_by_name(self.step)
            val = run_data.step_output_values[step_index]
        except (ValueError, IndexError):
            logger().warning(f"Output value for step {self.step} not found")
            return None
        return val


class Input(Reference):
    """A reference to a plan input."""

    name: str = Field(description="The name of the input.")

    def __init__(self, name: str) -> None:
        """Initialize the input."""
        self.name = name

    @override
    def get_legacy_name(self, plan: PortiaPlan) -> str:
        """Get the name of the reference to use with legacy Portia plans."""
        return self.name

    @override
    def get_value(self, run_data: RunData) -> ReferenceValue | None:
        """Get the value of the input."""
        plan_input = next(
            (_input for _input in run_data.plan.plan_inputs if _input.name == self.name), None
        )
        if not plan_input:
            logger().warning(f"Input {self.name} not found in plan")
            return None
        value = run_data.plan_run.plan_run_inputs.get(self.name)
        if not value:
            logger().warning(f"Value not found for input {self.name}")
            return None

        return ReferenceValue(
            value=value,
            description=plan_input.description or "Input to plan",
        )


class ReferenceValue(BaseModel):
    """Value that can be referenced."""

    value: Output = Field(description="The referenced value.")
    description: str = Field(description="Description of the referenced value.", default="")
