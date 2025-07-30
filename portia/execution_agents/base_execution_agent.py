"""Agents are responsible for executing steps of a PlanRun.

The BaseAgent class is the base class that all agents must extend.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import ToolMessage
from langgraph.graph import END, MessagesState

from portia.execution_agents.context import StepInput, build_context
from portia.execution_agents.execution_utils import (
    MAX_RETRIES,
    AgentNode,
    is_clarification,
    is_soft_tool_error,
)
from portia.execution_agents.output import LocalDataValue
from portia.logger import logger
from portia.plan import Plan, ReadOnlyStep, Step
from portia.plan_run import PlanRun, ReadOnlyPlanRun
from portia.telemetry.telemetry_service import ProductTelemetry

if TYPE_CHECKING:
    from portia.clarification import Clarification
    from portia.config import Config
    from portia.end_user import EndUser
    from portia.execution_agents.output import Output
    from portia.execution_hooks import ExecutionHooks
    from portia.storage import AgentMemory
    from portia.tool import Tool, ToolRunContext


class BaseExecutionAgent:
    """An ExecutionAgent is responsible for carrying out the task defined in the given Step.

    This BaseExecutionAgent is the class all ExecutionAgents must extend. Critically,
    ExecutionAgents must implement the execute_sync function which is responsible for
    actually carrying out the task as given in the step. They have access to copies of the
    step, plan_run and config but changes to those objects are forbidden.

    Optionally, new execution agents may also override the get_context function, which is
    responsible for building the system context for the agent. This should be done with
    thought, as the details of the system context are critically important for LLM
    performance.
    """

    def __init__(
        self,
        plan: Plan,
        plan_run: PlanRun,
        config: Config,
        end_user: EndUser,
        agent_memory: AgentMemory,
        tool: Tool | None = None,
        execution_hooks: ExecutionHooks | None = None,
    ) -> None:
        """Initialize the base agent with the given args.

        Importantly, the models here are frozen copies of those used by the Portia instance.
        They are meant as read-only references, useful for execution of the task
        but cannot be edited. The agent should return output via the response
        of the execute_sync method.

        Args:
            plan (Plan): The plan containing the steps.
            plan_run (PlanRun): The run that contains the step and related data.
            config (Config): The configuration settings for the agent.
            end_user (EndUser): The end user for the execution.
            agent_memory (AgentMemory): The agent memory for persisting outputs.
            tool (Tool | None): An optional tool associated with the agent (default is None).
            execution_hooks: Optional hooks for extending execution functionality.

        """
        self.plan = plan
        self.tool = tool
        self.config = config
        self.plan_run = plan_run
        self.end_user = end_user
        self.agent_memory = agent_memory
        self.execution_hooks = execution_hooks
        self.telemetry = ProductTelemetry()
        self.new_clarifications: list[Clarification] = []

    @property
    def step(self) -> Step:
        """Get the current step from the plan."""
        return self.plan.steps[self.plan_run.current_step_index]

    @abstractmethod
    def execute_sync(self) -> Output:
        """Run the core execution logic of the task synchronously.

        Implementation of this function is deferred to individual agent implementations,
        making it simple to write new ones.

        Returns:
            Output: The output of the task execution.

        """

    async def execute_async(self) -> Output:
        """Run the core execution logic of the task asynchronously.

        Implementation of this function is deferred to individual agent implementations,
        making it simple to write new ones. If not implemented, the agent will return a threaded
        version of the execute_sync method.

        Returns:
            Output: The output of the task execution.

        """
        return await asyncio.to_thread(self.execute_sync)

    def get_system_context(self, ctx: ToolRunContext, step_inputs: list[StepInput]) -> str:
        """Build a generic system context string from the step and run provided.

        This function retrieves the execution context and generates a system context
        based on the step and run provided to the agent.

        Args:
            ctx (ToolRunContext): The tool run ctx.
            step_inputs (list[StepInput]): The inputs for the step.

        Returns:
            str: A string containing the system context for the agent.

        """
        return build_context(
            ctx,
            self.plan_run,
            step_inputs,
        )

    def next_state_after_tool_call(
        self,
        config: Config,
        state: MessagesState,
        tool: Tool | None = None,
    ) -> Literal[AgentNode.TOOL_AGENT, AgentNode.SUMMARIZER, END]:  # type: ignore  # noqa: PGH003
        """Determine the next state after a tool call.

        This function checks the state after a tool call to determine if the run
        should proceed to the tool agent again, to the summarizer, or end.

        Args:
            config (Config): The configuration for the run.
            state (MessagesState): The current state of the messages.
            tool (Tool | None): The tool involved in the call, if any.

        Returns:
            Literal[AgentNode.TOOL_AGENT, AgentNode.SUMMARIZER, END]: The next state to transition
              to.

        Raises:
            ToolRetryError: If the tool has an error and the maximum retry limit has not been
              reached.

        """
        messages = state["messages"]
        last_message = messages[-1]
        errors = [msg for msg in messages if is_soft_tool_error(msg)]

        if is_soft_tool_error(last_message) and len(errors) < MAX_RETRIES:
            return AgentNode.TOOL_AGENT

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue

            if tool and self.execution_hooks and self.execution_hooks.after_tool_call:
                logger().debug("Calling after_tool_call execution hook")
                clarification = self.execution_hooks.after_tool_call(
                    tool,
                    message.content,
                    ReadOnlyPlanRun.from_plan_run(self.plan_run),
                    ReadOnlyStep.from_step(self.step),
                )
                logger().debug("Finished after_tool_call execution hook")
                if clarification:
                    self.new_clarifications.append(clarification)
                    return END

        # Prefers the step's structured output schema, if available.
        structured_output_schema = self.step.structured_output_schema or (
            tool and tool.structured_output_schema
        )

        if (
            "ToolSoftError" not in last_message.content
            and tool
            and (
                tool.should_summarize
                # If the value is larger than the threshold value, always summarise them as they are
                # too big to store the full value locally
                or config.exceeds_output_threshold(last_message.content)
                # If the tool has a structured output schema attached and hasn't already been
                # coerced to that schema, call the summarizer with structured response
                or (
                    structured_output_schema
                    and isinstance(last_message, ToolMessage)
                    and isinstance(last_message.artifact, LocalDataValue)
                    and not isinstance(last_message.artifact.value, structured_output_schema)
                )
            )
            and isinstance(last_message, ToolMessage)
            and not is_clarification(last_message.artifact)
        ):
            return AgentNode.SUMMARIZER
        return END
