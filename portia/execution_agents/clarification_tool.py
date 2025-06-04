"""Tool for raising clarifications if unsure on an arg."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia.clarification import InputClarification
from portia.tool import Tool, ToolRunContext


class ClarificationToolSchema(BaseModel):
    """Schema defining the inputs for the ClarificationTool."""

    argument_name: str = Field(
        description=(
            "The name of the argument that a value is needed for. This must match the "
            "argument name for the tool call exactly."
        ),
    )


class ClarificationTool(Tool[str]):
    """Raises a clarification if the agent is unsure of an argument."""

    id: str = "clarification_tool"
    name: str = "Clarification tool"
    description: str = (
        "Raises a clarification if you do not have enough information to provide a "
        "value for an argument."
    )
    args_schema: type[BaseModel] = ClarificationToolSchema
    output_schema: tuple[str, str] = ("str", "Model dump of the clarification to raise")
    step: int

    def run(self, ctx: ToolRunContext, argument_name: str) -> str:
        """Run the ClarificationTool."""
        return InputClarification(
            argument_name=argument_name,
            user_guidance=f"Missing Argument: {argument_name}",
            plan_run_id=ctx.plan_run.id,
            step=self.step,
            source="Clarification tool",
        ).model_dump_json()
