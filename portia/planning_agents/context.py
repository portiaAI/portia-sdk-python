"""Context helpers for PlanningAgents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from portia.templates.example_plans import DEFAULT_EXAMPLE_PLANS
from portia.templates.render import render_template

if TYPE_CHECKING:
    from portia.end_user import EndUser
    from portia.plan import Plan, PlanInput
    from portia.tool import Tool


def render_prompt_insert_defaults(
    query: str,
    tool_list: list[Tool],
    end_user: EndUser,
    examples: list[Plan] | None = None,
    plan_inputs: list[PlanInput] | None = None,
    previous_errors: list[str] | None = None,
) -> str:
    """Render the prompt for the PlanningAgent with defaults inserted if not provided."""
    system_context = default_query_system_context()
    non_default_examples_provided = True

    if examples is None:
        examples = DEFAULT_EXAMPLE_PLANS
        non_default_examples_provided = False
    tools_with_descriptions = get_tool_descriptions_for_tools(tool_list=tool_list)

    plan_input_dicts = None
    if plan_inputs:
        plan_input_dicts = [
            {"name": plan_input.name, "description": plan_input.description}
            for plan_input in plan_inputs
        ]

    return render_template(
        "default_planning_agent.xml.jinja",
        query=query,
        tools=tools_with_descriptions,
        end_user=end_user,
        examples=examples,
        system_context=system_context,
        non_default_examples_provided=non_default_examples_provided,
        plan_inputs=plan_input_dicts,
        previous_errors=previous_errors,
    )


def default_query_system_context() -> list[str]:
    """Return the default system context."""
    return [f"Today is {datetime.now(UTC).strftime('%Y-%m-%d')}"]


def get_tool_descriptions_for_tools(tool_list: list[Tool]) -> list[dict[str, str]]:
    """Given a list of tool names, return the descriptions of the tools."""
    return [
        {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "args": tool.args_schema.model_json_schema()["properties"],
            "output_schema": str(tool.output_schema),
        }
        for tool in tool_list
    ]
