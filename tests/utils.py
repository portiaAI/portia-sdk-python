"""Helpers to testing."""

from pydantic import BaseModel, Field

from portia.tool import Tool


class AdditionToolSchema(BaseModel):
    """Input for AdditionToolSchema."""

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
