"""Types to support Loops."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LoopBlock(BaseModel):
    """A loop block in the plan.

    This object is used to track the position of steps
    in the loop tree, if one is present.

    Args:
        start_step_index: The index of the first step in the loop.
        end_step_index: The index of the last step in the loop.

    """

    start_step_index: int = Field(description="The index of the first step in the loop.")
    end_step_index: int | None = Field(
        default=None, description="The index of the last step in the loop."
    )


class LoopBlockType(StrEnum):
    """The type of loop block."""

    START = "START"
    END = "END"


class LoopType(StrEnum):
    """The type of loop."""

    WHILE = "WHILE"
    DO_WHILE = "DO_WHILE"
    FOR_EACH = "FOR_EACH"


class LoopStepResult(BaseModel):
    """Output of a loop step.

    Args:
        type: The type of loop block that was executed.
        loop_result: The result of the loop predicate evaluation.
        start_loop_block_step_index: The step index of the start loop block (loop).
        end_loop_block_step_index: The step index of the end loop block (endloop).

    """

    block_type: LoopBlockType
    value: Any = Field(description="The value of the loop step.")
    loop_result: bool
    start_index: int
    end_index: int
