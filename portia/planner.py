"""Planner module creates plans from queries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field

from portia.plan import Plan
from portia.templates.example_plans import DEFAULT_EXAMPLE_PLANS
from portia.templates.render import render_template

if TYPE_CHECKING:
    from portia.config import Config
    from portia.tool import Tool
    from portia.llm_wrapper import BaseLLMWrapper
    from portia.tool_registry import ToolSet

logger = logging.getLogger(__name__)


class PlanOrError(BaseModel):
    """A plan or an error."""

    plan: Plan
    error: str | None = Field(
        default=None,
        description="An error message if the plan could not be created.",
    )


class Planner:
    """planner class."""

    def __init__(self, llm_wrapper: BaseLLMWrapper) -> None:
        """Init with the config."""
        self.llm_wrapper = llm_wrapper

    def generate_plan_or_error(
        self,
        query: str,
        tool_list: list[Tool],
        system_context: list[str] | None = None,
        examples: list[Plan] | None = None,
    ) -> PlanOrError:
        """Generate a plan or error using an LLM from a query and a list of tools."""
        prompt = _render_prompt_insert_defaults(
            query,
            tool_list,
            system_context,
            examples,
        )
        response = self.llm_wrapper.to_instructor(
            response_model=PlanOrError,
            messages=[
                {
                    "role": "system",
                    "content": "You are an outstanding task planner who can leverage many \
    tools as their disposal. Your job is provide a detailed plan of action to respond \
    to a user's prompt. When using multiple tools, pay attention to the arguments \
    that tools need to make sure the chain of calls works. If you are missing information do not \
    make up place holder variables like example@example.com. If you can't come up with a plan \
    provide a descriptive error instead - do not return plans with no steps.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        # don't use the ID assigned by the LLM but assign our own to ensure uniqueness.
        response.plan.id = uuid4()
        return response


def _render_prompt_insert_defaults(
    query: str,
    tool_list: list[Tool],
    system_context: list[str] | None = None,
    examples: list[Plan] | None = None,
) -> str:
    """Render the prompt for the query planner with defaults inserted if not provided."""
    if system_context is None:
        system_context = _default_query_system_context()

    if examples is None:
        examples = DEFAULT_EXAMPLE_PLANS

    tools_with_descriptions = _get_tool_descriptions_for_tools(tool_list=tool_list)

    return render_template(
        "query_planner.xml.jinja",
        query=query,
        tool_list=tools_with_descriptions,
        examples=examples,
        system_context=system_context,
    )


def _default_query_system_context() -> list[str]:
    """Return the default system context."""
    return [f"Today is {datetime.now(UTC).strftime('%Y-%m-%d')}"]


def _get_tool_descriptions_for_tools(tool_list: list[Tool]) -> list[dict[str, str]]:
    """Given a list of tool names, return the descriptions of the tools."""
    return [{"name": tool.name, "description": tool.description} for tool in tool_list]
