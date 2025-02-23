"""One shot planner is a single best effort attempt at planning based on the given query + tools."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langsmith import wrappers
from openai import OpenAI
from pydantic import BaseModel

from portia.execution_context import ExecutionContext, get_execution_context
from portia.llm_wrapper import LLMWrapper
from portia.open_source_tools.llm_tool import LLMTool
from portia.planners.context import (
    render_prompt_insert_defaults,
    render_prompt_insert_defaults_separated_tool_groups,
)
from portia.planners.planner import Planner, StepsOrError
from portia.templates.render import render_template
from portia.tool import Tool
from portia.tool_filtering_exploration.fake_tool import create_fake_tools

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Plan, Step
    from portia.tool import Tool

logger = logging.getLogger(__name__)


class ToolListResponse(BaseModel):
    """Response model for the tool list."""

    tool_ids: list[str]


class PlanJudgeResponse(BaseModel):
    """Response model for the plan judge."""

    plan_id: Literal["PLAN1", "PLAN2"]
    reason: str


class ThreeShotPlanner(Planner):
    """planner class."""

    def __init__(self, config: Config) -> None:
        """Init with the config."""
        self.llm_wrapper = LLMWrapper(config)

    def get_likely_tools(self, query: str, tool_list: list[Tool]) -> list[Tool]:
        """Get the likely tools for the query."""
        prompt = render_template(
            "tool_filterer.xml.jinja",
            query=query,
            tools=tool_list,
        )

        response = self.llm_wrapper.to_instructor(
            response_model=ToolListResponse,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at identifying the tools that are most likely to be "
                        "useful for a given query. You will be given a query and a list of tools. "
                        "You will need to return a list of the tool ids that are very likely to be "
                        "useful for the query. You should err on the side of too many tools rather "
                        "than too few."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        final_tools = [*response.tool_ids, LLMTool.LLM_TOOL_ID]

        # Filter the tool list based on the response
        filtered_tools = [tool for tool in tool_list if tool.id in final_tools]
        return filtered_tools

    def generate_steps_or_error(
        self,
        ctx: ExecutionContext,
        query: str,
        tool_list: list[Tool],
        examples: list[Plan] | None = None,
    ) -> StepsOrError:
        package_root = Path(__file__).parent.parent
        fake_tools_path = package_root / "tool_filtering_exploration" / "fake_tools_L.csv"
        extra_tools = create_fake_tools(str(fake_tools_path))
        final_tools = extra_tools + tool_list

        """Generate a plan or error using an LLM from a query and a list of tools."""
        likely_tools = self.get_likely_tools(query, final_tools)
        plan2 = self.sub_generate_steps_or_error(ctx, query, final_tools, examples)
        plan1 = self.sub_generate_steps_or_error(ctx, query, likely_tools, examples)
        plan3 = self.sub_generate_steps_or_error_separated_tool_groups(
            ctx, query, likely_tools, final_tools, examples
        )

        successful_plans = [plan for plan in [plan1, plan2, plan3] if not plan.error]

        tool_ids_used_in_plans = self._get_tools_used_in_plans(successful_plans)
        tools_used_in_plans = [t for t in final_tools if t.id in tool_ids_used_in_plans]

        prompt = render_template(
            "llm_plan_judge_three_shot.xml.jinja",
            query=query,
            plan1=plan1.model_dump(),
            plan2=plan2.model_dump(),
            plan3=plan3.model_dump(),
            tools=tools_used_in_plans,
        )

        client = wrappers.wrap_openai(OpenAI())

        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at judging the quality of plans. You will be given two "
                        "plans and a query. You will need to return the id of the plan that is "
                        "better suited to accomplish the query and a reason for your choie. "
                        "A good plan:\n"
                        "- would achieve the task goal (this is MOST important)\n"
                        "- does not use incorrect tools\n"
                        "- is concise \n"
                        "- is faithful to the original task request and does not hallucinate "
                        "information\n"
                        "Please assess which plan is the best and return your choice in JSON "
                        "format with the following schema:\n"
                        "{\n"
                        '  "plan_id": "PLAN1" | "PLAN2",\n'
                        '  "reason": "<reason>"\n'
                        "}\n"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        response_content = json.loads(response.choices[0].message.content or "")

        if "plan1" in response_content["plan_id"].lower():
            return plan1
        if "plan2" in response_content["plan_id"].lower():
            return plan2
        print("plan3 selected")
        return plan3

    def sub_generate_steps_or_error(
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

    def sub_generate_steps_or_error_separated_tool_groups(
        self,
        ctx: ExecutionContext,
        query: str,
        likely_tools: list[Tool],
        all_tools: list[Tool],
        examples: list[Plan] | None = None,
    ) -> StepsOrError:
        """Generate a plan or error using an LLM from a query and a list of tools."""
        ctx = get_execution_context()

        unlikely_tools = [tool for tool in all_tools if tool.id not in likely_tools]

        prompt = render_prompt_insert_defaults_separated_tool_groups(
            query,
            likely_tools,
            unlikely_tools,
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

        if not response.error:
            response.error = self._validate_tools_in_response(response.steps, all_tools)

        # Add LLMTool to the steps that don't have a tool_id.
        for step in response.steps:
            if step.tool_id is None:
                step.tool_id = LLMTool.LLM_TOOL_ID

        return StepsOrError(
            steps=response.steps,
            error=response.error,
        )

    def _get_tools_used_in_plans(self, plans: list[StepsOrError]) -> set[str]:
        """Get the tools used in the plans."""
        return {step.tool_id for plan in plans for step in plan.steps if step.tool_id}

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
