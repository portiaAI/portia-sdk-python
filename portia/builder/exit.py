"""Types to support exits."""

from pydantic import BaseModel, Field


class ExitStepResult(BaseModel):
    """Result of an ExitStep execution.

    This result indicates that the plan should exit gracefully.
    """

    message: str = Field(default="", description="The exit message to display.")
    error: bool = Field(
        default=False, description="Whether this exit represents an error condition."
    )
