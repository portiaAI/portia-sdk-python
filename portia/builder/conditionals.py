"""Types to support Conditionals."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ConditionalBlock(BaseModel):
    """A conditional block in the plan.

    This object is used to track the position of steps
    in the conditional tree, if one is present.

    Args:
        clause_step_indexes: The indexes of the conditional steps
            (i.e. the if_, else_if_, else_, endif steps).
        parent_conditional_block: The parent branch of this branch. If None,
            this is a root branch.

    """

    clause_step_indexes: list[int] = Field(default_factory=list)
    parent_conditional_block: "ConditionalBlock | None" = Field(
        default=None,
        description="The parent branch of this branch.",
    )


class ConditionalBlockClauseType(StrEnum):
    """The type of conditional block clause."""

    NEW_CONDITIONAL_BLOCK = "NEW_CONDITIONAL_BLOCK"
    ALTERNATE_CLAUSE = "ALTERNATE_CLAUSE"
    END_CONDITION_BLOCK = "END_CONDITION_BLOCK"


class ConditionalStepResult(BaseModel):
    """Output of a conditional step.

    Args:
        type: The type of conditional block clause that was executed.
        conditional_result: The result of the conditional predicate evaluation.
        next_clause_step_index: The step index of the next clause conditional to
            jump to if the conditional result is false.
        end_condition_block_step_index: The step index of the end condition block (endif).

    """

    type: ConditionalBlockClauseType
    conditional_result: bool
    next_clause_step_index: int
    end_condition_block_step_index: int
