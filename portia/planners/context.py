"""Context helpers for planners."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from portia.templates.example_plans import DEFAULT_EXAMPLE_PLANS
from portia.templates.render import render_template

if TYPE_CHECKING:
    from portia.plan import Plan
    from portia.tool import Tool


def render_prompt_insert_defaults(
    query: str,
    tool_list: list[Tool],
    system_context_extension: list[str] | None = None,
    examples: list[Plan] | None = None,
) -> str:
    """Render the prompt for the query planner with defaults inserted if not provided."""
    system_context = default_query_system_context(system_context_extension)

    if examples is None:
        examples = DEFAULT_EXAMPLE_PLANS

    tools_with_descriptions = get_tool_descriptions_for_tools(tool_list=tool_list)

    return render_template(
        "query_planner.xml.jinja",
        query=query,
        tools=tools_with_descriptions,
        examples=examples,
        system_context=system_context,
    )


def default_query_system_context(
    system_context_extension: list[str] | None = None,
) -> list[str]:
    """Return the default system context."""
    base_context = [f"Today is {datetime.now(UTC).strftime('%Y-%m-%d')}"]
    if system_context_extension:
        base_context.extend(system_context_extension)
    return base_context


def get_tool_descriptions_for_tools(tool_list: list[Tool]) -> list[dict[str, str]]:
    """Given a list of tool names, return the descriptions of the tools."""
    return [
        {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "args": tool.args_schema.model_json_schema()["properties"],
        }
        for tool in tool_list
    ]
