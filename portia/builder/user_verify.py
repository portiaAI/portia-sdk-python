"""Implementation of the various step types used in :class:`PlanV2`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from portia.builder.step import Step

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import Field

from portia.clarification import (
    ClarificationCategory,
    UserVerificationClarification,
)
from portia.errors import PlanRunExitError
from portia.plan import Step as StepData

if TYPE_CHECKING:
    from portia.builder.plan import Plan
    from portia.run_context import RunContext


class UserVerifyStep(Step):
    """A step that requests user confirmation before proceeding with plan execution.

    This step pauses execution to ask the user to verify or approve a message.
    If the user rejects the verification, the plan execution will stop with an error.

    This pauses plan execution and asks the user to confirm or reject the provided
    message. The plan will only continue if the user confirms. If the user rejects,
    the plan execution will stop with an error. This is useful for getting user approval before
    taking important actions like sending emails, making purchases, or modifying data.

    A UserVerificationClarification is used to get the verification from the user, so ensure you
    have set up handling for this type of clarification in order to use this step. For more
    details, see https://docs.portialabs.ai/understand-clarifications.

    This step outputs True if the user confirms.
    """

    message: str = Field(
        description="The message or action requiring user verification/approval. "
        "It can include references to previous step outputs or plan inputs "
        "(using Input / StepOutput references)"
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        return f"UserVerifyStep(message='{self.message}')"

    @override
    @traceable(name="User Verify Step - Run")
    async def run(self, run_data: RunContext) -> bool | UserVerificationClarification:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Prompt the user for confirmation.

        Returns a UserVerificationClarification to get input from the user (if not already
        provided).

        If the user has already confirmed, returns True. Otherwise, if the user has rejected the
        verification, raises a PlanRunExitError.
        """
        message = self._template_references(self.message, run_data)

        previous_clarification = run_data.plan_run.get_clarification_for_step(
            ClarificationCategory.USER_VERIFICATION
        )

        if not previous_clarification or not previous_clarification.resolved:
            return UserVerificationClarification(
                plan_run_id=run_data.plan_run.id,
                user_guidance=str(message),
                source="User verify step",
            )

        if previous_clarification.response is False:
            raise PlanRunExitError(f"User rejected verification: {message}")

        return True

    @override
    def to_step_data(self, plan: Plan) -> StepData:
        """Convert this UserVerifyStep to a legacy Step."""
        return Step(
            task=f"User verification: {self.message}",
            inputs=[],
            tool_id=None,
            output=plan.step_output_name(self),
            condition=self._get_legacy_condition(plan),
        )
