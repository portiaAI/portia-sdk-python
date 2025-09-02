"""Tool for raising clarifications if unsure on an arg."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia.clarification import InputClarification
from portia.tool import Tool, ToolRunContext


class ReActClarificationToolSchema(BaseModel):
    """Schema defining the inputs for the ClarificationTool."""

    guidance: str = Field(
        description=("The guidance that is needed from the users."),
    )


class ReActClarificationTool(Tool[InputClarification]):
    """Raises a clarification if the agent does not have enough information to proceed."""

    id: str = "clarification_tool"
    name: str = "Clarification tool"
    description: str = (
        "Raises a clarification with the user if you do not have enough information to proceed"
    )
    args_schema: type[BaseModel] = ReActClarificationToolSchema
    output_schema: tuple[str, str] = ("InputClarification", "The clarification to raise")

    def run(self, ctx: ToolRunContext, guidance: str) -> InputClarification:
        """Run the ReActClarificationTool."""
        return InputClarification(
            argument_name="react_agent_clarification",
            user_guidance=guidance,
            plan_run_id=ctx.plan_run.id,
            source="ReActClarificationTool",
        )
