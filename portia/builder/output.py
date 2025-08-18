from __future__ import annotations

from pydantic import BaseModel

from portia.execution_agents.output import Output


def default_step_name(step_index: int) -> str:
    """Return the default name for the step."""
    return f"step_{step_index}"


class StepOutput(BaseModel):
    """A reference to the output of a step."""

    step: str

    def __init__(self, step: str | int) -> None:
        """Initialize the step output.

        Args:
            step: The step to reference the output of. If a string is provided, this will be used
              to find the step by name. If an integer is provided, this will be used to find the
              step by index.

        """
        if isinstance(step, int):
            step = default_step_name(step)
        super().__init__(step=step)

    def __hash__(self) -> int:
        """Hash the step output."""
        return hash(self.step)

    def __eq__(self, other: object) -> bool:
        """Check if the step output is equal to another object."""
        if not isinstance(other, StepOutput):
            return NotImplemented
        return self.step == other.step


class StepOutputValue(BaseModel):
    """Value of a step output."""

    value: Output
    description: str
