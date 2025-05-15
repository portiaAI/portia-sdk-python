"""DefaultPlanningAgent is a single best effort attempt at planning based on the given query + tools."""  # noqa: E501

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from portia.model import Message
from portia.open_source_tools.llm_tool import LLMTool
from portia.planning_agents.base_planning_agent import BasePlanningAgent, StepsOrError
from portia.planning_agents.context import render_prompt_insert_defaults

if TYPE_CHECKING:
    from portia.config import Config
    from portia.end_user import EndUser
    from portia.plan import Plan, PlanInput, Step
    from portia.tool import Tool

logger = logging.getLogger(__name__)


class DefaultPlanningAgent(BasePlanningAgent):
    """DefaultPlanningAgent class."""

    def __init__(self, config: Config) -> None:
        """Init with the config."""
        self.model = config.get_planning_model()

    def generate_steps_or_error(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        examples: list[Plan] | None = None,
        plan_inputs: list[PlanInput] | None = None,
    ) -> StepsOrError:
        """Generate a plan or error using an LLM from a query and a list of tools."""
        retries_remaining = 2
        previous_errors = []

        while retries_remaining >= 0:
            prompt = render_prompt_insert_defaults(
                query,
                tool_list,
                end_user,
                examples,
                plan_inputs,
                previous_errors,
            )
            response = self.model.get_structured_response(
                schema=StepsOrError,
                messages=[
                    Message(
                        role="system",
                        content="""
You are an outstanding task planner who can leverage many tools at their disposal. Your job is
to provide a detailed plan of action in the form of a set of steps to respond to a user's prompt.

IMPORTANT GUIDELINES:
- When using multiple tools, pay attention to the  tools to make sure the chain of steps works,
 but DO NOT provide any examples or assumptions  in the task descriptions.
- If you are missing information do not  make up placeholder variables like example@example.com.
- When creating the description for a step of the plan, if you need information from the previous
 step, DO NOT guess what that step will produce - instead, specify the previous step's output as an
 input for this step and allow this to be handled when we execute the plan.
- If you can't come up with a plan provide a descriptive error instead - DO NOT
 create a plan with zero steps. When returning an error, return only the JSON object with
 the error message filled in plus an empty list of steps - DO NOT include text (explaining the
 error or otherwise) outside the JSON object.
- Information provided in the EndUser block will also be provided at execution so you do not need to
 add inputs for this information. However if information about the EndUser is not available now
 it will also NOT be available later and you should return an error if it is required and there is
 no other way to retrieve the information.
- For EVERY tool that requires an id as an input, make sure to check
 if there's a corresponding tool call that provides the id from natural language if possible.
 For example, if a tool asks for a user ID check if there's a tool call that provides
 the user IDs before making the tool call that  requires the user ID.
- For conditional steps:
  1. Task field: Write only the task description without conditions.
  2. Condition field: Write the condition in concise natural language.
- Do not use the condition field for non-conditional steps.
- If plan inputs are provided, make sure you specify them as inputs to the appropriate steps.
- Only use plan inputs if they are provided - DO NOT make any up
                    """,
                    ),
                    Message(role="user", content=prompt),
                ],
            )

            # Check for errors in the response
            if response.error:
                # We don't retry LLM errors as we have no new useful information to provide
                return StepsOrError(
                    steps=response.steps,
                    error=response.error,
                )

            tool_error = self._validate_tools_in_response(response.steps, tool_list)
            if tool_error:
                previous_errors.append(
                    StepsOrError(
                        steps=response.steps,
                        error=f"Attempt {3-retries_remaining}: {tool_error}",
                    )
                )
                retries_remaining -= 1
                continue

            input_error = self._validate_inputs_in_response(response.steps, plan_inputs)
            if input_error:
                previous_errors.append(
                    StepsOrError(
                        steps=response.steps,
                        error=f"Attempt {3-retries_remaining}: {input_error}",
                    )
                )
                retries_remaining -= 1
                continue

            # Add LLMTool to the steps that don't have a tool_id
            for step in response.steps:
                if step.tool_id is None:
                    step.tool_id = LLMTool.LLM_TOOL_ID

            return StepsOrError(
                steps=response.steps,
                error=None,
            )

        # If we get here, we've exhausted all retries
        return StepsOrError(
            steps=[],
            error="\n".join(str(error) for error in previous_errors),
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

    def _validate_inputs_in_response(
        self,
        steps: list[Step],
        plan_inputs: list[PlanInput] | None = None,
    ) -> str | None:
        """Validate that each step's inputs are either from plan inputs or previous step outputs.

        Args:
            steps (list[Step]): List of steps from the response
            plan_inputs (list[PlanInput] | None): Optional list of plan inputs

        Returns:
            Error message if inputs are invalid, None otherwise

        """
        plan_inputs_names = {input_.name for input_ in (plan_inputs or [])}
        step_outputs = set()

        for i, step in enumerate(steps):
            for input_var in step.inputs:
                if input_var.name not in plan_inputs_names and input_var.name not in step_outputs:
                    return (
                        f"Step {i+1} uses input '{input_var.name}' which is neither a plan input "
                        f"nor an output from a previous step"
                    )

            if step.output:
                step_outputs.add(step.output)

        return None
