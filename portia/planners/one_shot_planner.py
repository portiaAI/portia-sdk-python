"""Planner module creates plans from queries."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from portia.context import ExecutionContext, get_execution_context
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan, PlanContext, Step
from portia.planners.planner import PlanOrError, Planner, StepsOrError
from portia.templates.example_plans import DEFAULT_EXAMPLE_PLANS
from portia.templates.render import render_template

if TYPE_CHECKING:
    from portia.config import Config
    from portia.tool import Tool

logger = logging.getLogger(__name__)


class OneShotPlanner(Planner):
    """planner class."""

    def __init__(self, config: Config) -> None:
        """Init with the config."""
        self.llm_wrapper = LLMWrapper(config)

    def generate_plan_or_error(
        self,
        ctx: ExecutionContext,
        query: str,
        tool_list: list[Tool],
        examples: list[Plan] | None = None,
    ) -> PlanOrError:
        """Generate a plan or error using an LLM from a query and a list of tools."""
        ctx = get_execution_context()
        prompt = _render_prompt_insert_defaults(
            query,
            tool_list,
            ctx.planner_system_context_extension,
            examples,
        )
        response = self.llm_wrapper.to_instructor(
            response_model=StepsOrError,
            messages=[
                {
                    "role": "system",
                    "content": "You are an outstanding task planner who can leverage many \
    tools as their disposal. Your job is provide a detailed plan of action in the form of a set of \
    steps to respond to a user's prompt. When using multiple tools, pay attention to the arguments \
    that tools need to make sure the chain of calls works. If you are missing information do not \
    make up placeholder variables like example@example.com. If you can't come up with a plan \
    provide a descriptive error instead - do not return plans with no steps.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return PlanOrError(
            plan=Plan(
                plan_context=PlanContext(
                    query=query,
                    tool_ids=[tool.id for tool in tool_list],
                ),
                steps=response.steps,
            ),
            error=response.error,
        )


def _render_prompt_insert_defaults(
    query: str,
    tool_list: list[Tool],
    system_context_extension: list[str] | None = None,
    examples: list[Plan] | None = None,
) -> str:
    """Render the prompt for the query planner with defaults inserted if not provided."""
    system_context = _default_query_system_context(system_context_extension)

    if examples is None:
        examples = DEFAULT_EXAMPLE_PLANS

    tools_with_descriptions = _get_tool_descriptions_for_tools(tool_list=tool_list)

    return render_template(
        "query_planner.xml.jinja",
        query=query,
        tools=tools_with_descriptions,
        examples=examples,
        system_context=system_context,
    )


def _default_query_system_context(
    system_context_extension: list[str] | None = None,
) -> list[str]:
    """Return the default system context."""
    base_context = [f"Today is {datetime.now(UTC).strftime('%Y-%m-%d')}"]
    if system_context_extension:
        base_context.extend(system_context_extension)
    return base_context


def _get_tool_descriptions_for_tools(tool_list: list[Tool]) -> list[dict[str, str]]:
    """Given a list of tool names, return the descriptions of the tools."""
    return [
        {"id": tool.id, "name": tool.name, "description": tool.description} for tool in tool_list
    ]
