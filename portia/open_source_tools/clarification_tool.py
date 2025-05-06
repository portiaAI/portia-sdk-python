"""Tool for raising clarifications if unsure on an arg."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia.clarification import Clarification, InputClarification
from portia.tool import Tool, ToolRunContext


class ClarificationToolSchema(BaseModel):
    """Schema defining the inputs for the ClarificationTool."""

    argument_name: str = Field(
        description="The name of the argument that a value is needed for.",
    )


class ClarificationTool(Tool[Clarification]):
    """Raises a clarification if we are unsure of an argument."""

    id: str = "clarification_tool"
    name: str = "Clarification tool"
    description: str = "Raises a clarification if we are unsure of an argument."
    args_schema: type[BaseModel] = ClarificationToolSchema
    output_schema: tuple[str, str] = ("str", "Model dump of the clarification to raise")

    def run(self, ctx: ToolRunContext, argument_name: str) -> Clarification:
        """Run the FileReaderTool."""
        return InputClarification(
            argument_name=argument_name,
            user_guidance=f"Missing Argument: {argument_name}",
            plan_run_id=ctx.plan_run_id,
        ).model_dump_json()
