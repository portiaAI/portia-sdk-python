"""Implementation of the user input step."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from portia.builder.step_v2 import StepV2

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import Field

from portia.clarification import (
    ClarificationCategory,
    ClarificationType,
    InputClarification,
    MultipleChoiceClarification,
)
from portia.plan import Step

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.run_context import RunContext


class UserInputStep(StepV2):
    """A step that requests input from the user and returns their response.

    This pauses plan execution and prompts the user to provide input. If options are
    provided, the user must choose from the given choices (multiple choice). If no
    options are provided, the user can enter free-form text.

    A Clarification (either InputClarification or MultipleChoiceClarification) is used to get
    the input from the user, so ensure you have set up handling for the required type of
    clarification in order to use this step. For more details, see
    https://docs.portialabs.ai/understand-clarifications.

    The user's response becomes the output of this step and can be referenced by
    subsequent steps in the plan.
    """

    message: str = Field(description="The prompt or question to display to the user.")
    options: list[Any] | None = Field(
        default=None,
        description=(
            "Available choices for multiple-choice input. If provided, the user must select "
            "from these options. If None, allows free-form text input. Options can include "
            "references to previous step outputs or plan inputs (using Input / StepOutput "
            "references)"
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        input_type = "multiple choice" if self.options else "text input"
        return f"UserInputStep(type='{input_type}', message='{self.message}')"

    def _create_clarification(self, run_data: RunContext) -> ClarificationType:
        """Create the appropriate clarification based on whether options are provided."""
        resolved_message = self._template_references(self.message, run_data)

        if self.options:
            options = [self._resolve_references(o, run_data) for o in self.options]
            return MultipleChoiceClarification(
                plan_run_id=run_data.plan_run.id,
                user_guidance=str(resolved_message),
                options=options,
                argument_name=run_data.plan.step_output_name(self),
                source="User input step",
            )
        return InputClarification(
            plan_run_id=run_data.plan_run.id,
            user_guidance=str(resolved_message),
            argument_name=run_data.plan.step_output_name(self),
            source="User input step",
        )

    @override
    @traceable(name="User Input Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Request input from the user and return the response."""
        clarification_type = (
            ClarificationCategory.MULTIPLE_CHOICE if self.options else ClarificationCategory.INPUT
        )

        previous_clarification = run_data.plan_run.get_clarification_for_step(clarification_type)

        if not previous_clarification or not previous_clarification.resolved:
            return self._create_clarification(run_data)

        return previous_clarification.response

    @override
    def to_step_data(self, plan: PlanV2) -> Step:
        """Convert this UserInputStep to a legacy Step."""
        input_type = "Multiple choice" if self.options else "Text input"
        return Step(
            task=f"User input ({input_type}): {self.message}",
            inputs=[],
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
