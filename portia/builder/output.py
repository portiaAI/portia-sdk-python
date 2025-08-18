"""@@@TODO."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia.execution_agents.output import Output
from portia.logger import logger


def default_step_name(step_index: int) -> str:
    """Return the default name for the step."""
    return f"step_{step_index}"


class StepOutput(BaseModel):
    """A reference to the output of a step."""

    step: str | int = Field(
        description="The step to reference the output of. If a string is provided, this will be used"
        "to find the step by name. If an integer is provided, this will be used to find the"
        "step by index (steps are 0-indexed)."
    )

    def __init__(self, step: str | int) -> None:
        """Initialize the step output."""
        super().__init__(step=step)

    def get_value(self, step_output_values: list[StepOutputValue]) -> StepOutputValue | None:
        """Get the value of the step output."""
        if isinstance(self.step, int):
            return step_output_values[self.step]
        for output in step_output_values:
            if output.step_name == self.step:
                return output
        logger().warning(f"Output from step {self.step} not found")
        return None


class StepOutputValue(BaseModel):
    """Value of a step output."""

    step_name: str
    value: Output
    description: str
