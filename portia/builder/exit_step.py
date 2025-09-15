"""Types to support exits."""

import sys

from portia.builder.exit import ExitStepResult

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import Field

from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import StepV2
from portia.plan import Step
from portia.run_context import RunContext


class ExitStep(StepV2):
    """A step that causes the plan to exit gracefully.

    This step allows for early termination of a plan with an optional message and error flag. When
    executed, the plan will stop execution and return the specified output.
    """

    message: str = Field(default="", description="The message to include when exiting the plan.")
    error: bool = Field(
        default=False, description="Whether this exit represents an error condition."
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        error_indicator = " (ERROR)" if self.error else ""
        message_info = f" - {self.message}" if self.message else ""
        return f"ExitStep{error_indicator}{message_info}"

    @override
    @traceable(name="Exit Step - Run")
    async def run(self, run_data: RunContext) -> ExitStepResult:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Execute the exit step, causing the plan to terminate gracefully."""
        message = self._template_references(self.message, run_data)
        return ExitStepResult(message=message, error=self.error)

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this ExitStep to a legacy Step."""
        if self.error:
            task = f"Exit plan with error: {self.message}"
        else:
            task = f"Exit plan: {self.message}"

        return Step(
            task=task,
            inputs=[],
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
