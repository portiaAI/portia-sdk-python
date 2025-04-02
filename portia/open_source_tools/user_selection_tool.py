from __future__ import annotations  # noqa: D100

from pydantic import BaseModel

from portia.clarification import Clarification, MultipleChoiceClarification
from portia.tool import Tool, ToolRunContext


class UserSelectionToolSchema(BaseModel):
    """Input for UserSelectionTool."""

    options: list[str]
    chosen_value: str | None



class UserSelectionTool(Tool[str]):
    """Creates a Clarification for a user to select something."""

    id: str = "user_selection_tool"
    name: str = "User Selection Tool"
    description: str = "Tool to get a user's choice from a list of options. You can use this tool with any list of options for for any multiple choice question."
    args_schema: type[BaseModel] = UserSelectionToolSchema
    output_schema: tuple[str, str] = (
        "Clarification | str",
        "Clarification: The value of the Clarification or str: The value chosen.",
    )

    def run(
        self,
        ctx: ToolRunContext,
        options: list[str],
        chosen_value: str | None,
    ) -> Clarification | str:
        """Run the UserSelectionTool."""
        if not chosen_value or chosen_value == "":
            return MultipleChoiceClarification(
                plan_run_id=ctx.plan_run_id,
                user_guidance="Please select an option",
                argument_name="chosen_value",
                options=options,
            )
        return chosen_value
