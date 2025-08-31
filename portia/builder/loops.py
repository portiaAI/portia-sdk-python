"""Types to support Loops."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class LoopBlock(BaseModel):
    """A loop block in the plan.

    This object is used to track the position of steps
    in the loop tree, if one is present.

    Args:
        start_step_index: The index of the first step in the loop.
        end_step_index: The index of the last step in the loop.

    """
    start_step_index: int | None = Field(description="The index of the first step in the loop.") 
    end_step_index: int | None = Field(description="The index of the last step in the loop.")


    @model_validator(mode="after")
    def validate_start_end_step_indexes(self) -> Self:
        """Validate the start and end step indexes."""
        if self.start_step_index is None and self.end_step_index is None:
            raise ValueError("Start and end step indexes cannot both be None")
        return self

class LoopBlockClauseType(StrEnum):
    """The type of loop block clause."""

    START = "START"
    END = "END"