"""DefaultPlanningAgent is a single best effort attempt at planning based on the given query + tools."""  # noqa: E501

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from portia.config import PLANNING_MODEL_KEY
from portia.execution_context import ExecutionContext, get_execution_context
from portia.llm_wrapper import LLMWrapper
from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool
from portia.planning_agents.base_planning_agent import BasePlanningAgent, StepsOrError
from portia.planning_agents.context import render_prompt_insert_defaults

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Plan, Step
    from portia.tool import Tool

logger = logging.getLogger(__name__)


class DefaultPlanningAgent(BasePlanningAgent):
    """DefaultPlanningAgent class."""

    def __init__(self, config: Config) -> None:
        """Init with the config."""
        self.llm_wrapper = LLMWrapper.for_usage(PLANNING_MODEL_KEY, config)

    def generate_steps_or_error(
        self,
        ctx: ExecutionContext,
        query: str,
        tool_list: list[Tool],
        examples: list[Plan] | None = None,
    ) -> StepsOrError:
        """Generate a plan or error using an LLM from a query and a list of tools."""
        ctx = get_execution_context()
        prompt = render_prompt_insert_defaults(
            query,
            tool_list,
            ctx.planning_agent_system_context_extension,
            examples,
        )
        response = self.llm_wrapper.to_instructor(
            response_model=StepsOrError,
            messages=[
                Message(
                    role="system",
                    content="You are an outstanding task planner who can leverage many \
    tools as their disposal. Your job is provide a detailed plan of action in the form of a set of \
    steps to respond to a user's prompt. When using multiple tools, pay attention to the  \
    tools to make sure the chain of steps works, but DO NOT provide any examples or assumptions \
    in the task descriptions. If you are missing information do not \
    make up placeholder variables like example@example.com. If you can't come up with a plan \
    provide a descriptive error instead - do not return plans with no steps. For EVERY tool that \
    requires an id as an input, make sure to check if there's a corresponding tool call that\
    provides the id from natural language if possible. For example, if a tool asks for a user ID\
    check if there's a tool call that provides the user IDs before making the tool call that \
    requires the user ID. For conditional steps: \
    1. Task field: Write only the task description without conditions. \
    2. Condition field: Write the condition in concise natural language. \
    Do not use the condition field for non-conditional steps.",
                ),
                Message(role="user", content=prompt),
            ],
        )

        if not response.error:
            response.error = self._validate_tools_in_response(response.steps, tool_list)

        # Add LLMTool to the steps that don't have a tool_id.
        for step in response.steps:
            if step.tool_id is None:
                step.tool_id = LLMTool.LLM_TOOL_ID

        return StepsOrError(
            steps=response.steps,
            error=response.error,
        )

    def _validate_tools_in_response(self, steps: list[Step], tool_list: list[Tool]) -> str | None:
        """Validate that all tools in the response steps exist in the provided tool list.

        Args:
            steps (list[Step]): List of steps from the response
            tool_list (list[Tool]): List of available tools

        Returns:
            Error message if tools are missing, None otherwise

        """
        tool_ids = [tool.id for tool in tool_list]
        missing_tools = [
            step.tool_id for step in steps if step.tool_id and step.tool_id not in tool_ids
        ]
        return (
            f"Missing tools {', '.join(missing_tools)} from the provided tool_list"
            if missing_tools
            else None
        )
