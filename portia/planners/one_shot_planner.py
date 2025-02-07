"""One shot planner is a single best effort attempt at planning based on the given query + tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from portia.execution_context import ExecutionContext, get_execution_context
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan, PlanContext
from portia.planners.context import render_prompt_insert_defaults
from portia.planners.planner import Planner, PlanOrError, StepsOrError

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
        prompt = render_prompt_insert_defaults(
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
    provide a descriptive error instead - do not return plans with no steps. For EVERY tool that \
    requires an id as an input, make sure to check if there's a corresponding tool call that\
    provides the id from natural language if possible. For example, if a tool asks for a user ID\
    check if there's a tool call that provides the user IDs before making the tool call that \
    requires the user ID.",
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
