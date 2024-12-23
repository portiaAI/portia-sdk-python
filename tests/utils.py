"""Helpers to testing."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia.clarification import Clarification, InputClarification
from portia.tool import Tool, ToolHardError, ToolSoftError


class AdditionToolSchema(BaseModel):
    """Input for AdditionTool."""

    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")


class AdditionTool(Tool):
    """Adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("int", "int: The value of the addition")

    def run(self, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b


class ClarificationToolSchema(BaseModel):
    """Input for ClarificationTool."""

    user_guidance: str
    raise_clarification: bool


class ClarificationTool(Tool):
    """Returns a Clarification."""

    id: str = "clarification_tool"
    name: str = "Clarification Tool"
    description: str = "Returns a clarification"
    args_schema: type[BaseModel] = ClarificationToolSchema
    output_schema: tuple[str, str] = (
        "Clarification",
        "Clarification: The value of the Clarification",
    )

    def run(self, user_guidance: str, raise_clarification: bool) -> Clarification | None:  # noqa: FBT001
        """Add the numbers."""
        if raise_clarification:
            return InputClarification(
                user_guidance=user_guidance,
                argument_name="raise_clarification",
            )
        return None


class MockToolSchema(BaseModel):
    """Input for MockTool."""


class MockTool(Tool):
    """A mock tool class for testing purposes."""

    id: str = "mock_tool"
    description: str = "do nothing"
    args_schema: type[BaseModel] = MockToolSchema
    output_schema: tuple[str, str] = ("None", "None: returns nothing")

    def run(self) -> None:
        """Do nothing."""
        return


class ErrorToolSchema(BaseModel):
    """Input for ErrorTool."""

    error_str: str
    return_soft_error: bool


class ErrorTool(Tool):
    """Returns an Error."""

    id: str = "error_tool"
    name: str = "Error Tool"
    description: str = "Returns a error"
    args_schema: type[BaseModel] = ErrorToolSchema
    output_schema: tuple[str, str] = (
        "Error",
        "Error: The value of the error",
    )

    def run(self, error_str: str, return_soft_error: bool) -> None:  # noqa: FBT001
        """Return the error."""
        if return_soft_error:
            raise ToolSoftError(error_str)
        raise ToolHardError(error_str)
