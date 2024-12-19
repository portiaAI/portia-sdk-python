"""Test harness."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia import InMemoryToolRegistry
from portia.clarification import InputClarification
from portia.config import Config
from portia.runner import Runner
from portia.tool import Tool
from portia.workflow import WorkflowState


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

    def run(self, a: int, b: int) -> int | InputClarification:
        """Add the numbers."""
        if a == 1:
            return InputClarification(
                user_guidance="Please help me to solve",
                argument_name="a",
            )
        return a + b


registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
runner = Runner(Config(), tool_registry=registry)


output = runner.run_query("Add 1 + 2", tools=["add_tool"])

while output.state != WorkflowState.COMPLETE:
    for clarification in output.get_outstanding_clarifications():
        # resolve clarification
        clarification.resolve(10)

    # after we've resovled all clars we resume
    runner.resume_workflow(output)


print(output)
