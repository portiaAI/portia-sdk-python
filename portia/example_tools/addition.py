"""Simple Addition Tool."""

from pydantic import BaseModel, Field

from portia.tool import Tool


class AdditionToolSchema(BaseModel):
    """Input for AdditionTool."""

    a: float = Field(..., description="The first number to add")
    b: float = Field(..., description="The second number to add")


class AdditionTool(Tool):
    """Adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two float and adds them together"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("float", "float: The value of the addition")

    def run(self, a: float, b: float) -> float:
        """Add the numbers."""
        return a + b
