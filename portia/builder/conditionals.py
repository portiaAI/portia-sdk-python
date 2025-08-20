"""Types to support Conditionals."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ConditionalBranch(BaseModel):
    """A conditional branch in the plan.

    This object is used to track the position of steps
    in the conditional tree, if one is present.

    Args:
        branch_step_indexes: The indexes of the conditional steps
            (i.e. the if_, else_if_, else_, endif steps).
        parent_branch: The parent branch of this branch. If None,
            this is a root branch.

    """

    branch_step_indexes: list[int] = Field(default_factory=list)
    parent_branch: "ConditionalBranch | None" = Field(
        default=None,
        description="The parent branch of this branch.",
    )


class BranchStateType(StrEnum):
    """The type of branch state."""

    ENTER_BRANCH = "ENTER_BRANCH"
    ALTERNATE_BRANCH = "ALTERNATE_BRANCH"
    EXIT_BRANCH = "EXIT_BRANCH"


class ConditionalStepResult(BaseModel):
    """Output of a conditional step.

    Args:
        type: The type of Conditional node that was executed.
        conditional_result: The result of the conditional predicate evaluation.
        next_branch_step_index: The step index of the next branch conditional to
            jump to if the conditional result is false.
        branch_exit_step_index: The step index of the branch exit step (endif).

    """

    type: BranchStateType
    conditional_result: bool
    next_branch_step_index: int
    branch_exit_step_index: int
