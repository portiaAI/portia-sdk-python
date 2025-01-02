"""Test harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from portia.config import StorageClass, default_config
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import InMemoryToolRegistry, PortiaToolRegistry
from portia.workflow import WorkflowState

if TYPE_CHECKING:
    from portia.clarification import InputClarification


class AdditionToolSchema(BaseModel):
    """Input for AdditionToolSchema."""

    a: float = Field(..., description="The first number to add")
    b: float = Field(..., description="The second number to add")


class AdditionTool(Tool):
    """Adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("int", "int: The value of the addition")

    def run(self, a: float, b: float) -> float | InputClarification:
        """Add the numbers."""
        return a + b


config = default_config()
config.storage_class = StorageClass.CLOUD

local_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
cloud_registry = PortiaToolRegistry(
    config=config,
)
registry = local_registry + cloud_registry

runner = Runner(
    config,
    tool_registry=registry,
)


output = runner.run_query(
    "Get the temperature in London and Sydney and then add the two temperatures together.",
)

# optional clarification resolution block
while output.state == WorkflowState.NEED_CLARIFICATION:
    for clarification in output.get_outstanding_clarifications():  # noqa: B007
        # resolve clarification
        continue

    # after we've resolved all clarifications we resume
    runner.resume_workflow(output)


print(output)  # noqa: T201
