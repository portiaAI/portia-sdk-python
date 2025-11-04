"""Implementation of the ReAct agent step."""

from __future__ import annotations

import sys
from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Any

from portia.builder.step import Step
from portia.model import GenerativeModel  # noqa: TC001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.execution_agents.react_agent import ReActAgent
from portia.plan import Step as StepData
from portia.tool import Tool  # noqa: TC001
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.plan import Plan
    from portia.run_context import RunContext


class ReActAgentStep(Step):
    """A step where an LLM agent uses ReAct reasoning to complete a task with multiple tools.

    Unlike SingleToolAgentStep which is limited to one specific tool and one tool call, this step
    allows an LLM agent to reason about which tools to use and when to use them. The agent
    follows the ReAct (Reasoning and Acting) pattern, iteratively thinking about the
    problem and taking actions until the task is complete.
    """

    task: str = Field(description="Natural language description of the task to accomplish.")
    tools: Sequence[str | Tool] = Field(
        description=(
            "IDs of the tools the agent can use to complete the task or Tool instances themselves."
        )
    )
    inputs: list[Any] = Field(
        default_factory=list,
        description=(
            "The inputs for the task. The inputs can be references to previous step outputs / "
            "plan inputs (using StepOutput / Input) or just plain values. They are passed in as "
            "additional context to the agent when it is completing the task."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class defining the expected structure of the agent's output. "
            "If provided, the output from the agent will be coerced to match this schema."
        ),
    )
    tool_call_limit: int = Field(
        default=25,
        description="The maximum number of tool calls to make before the agent stops.",
    )
    allow_agent_clarifications: bool = Field(
        default=False,
        description=(
            "Whether to allow the agent to ask clarifying questions to the user "
            "if it is unable to proceed. When set to true, the agent can output clarifications "
            "that can be resolved by the user to get input. In order to use this, make sure you "
            "clarification handler set up that is capable of handling InputClarifications."
        ),
    )
    model: GenerativeModel | str | None = Field(
        default=None,
        description=(
            "The model to use for this agent. If not provided, the planning model from the config "
            "will be used."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        tools_list = [t if isinstance(t, str) else t.id for t in self.tools]
        return f"ReActAgentStep(task='{self.task}', tools='{tools_list}', {output_info})"

    @override
    @traceable(name="ReAct Agent Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent step."""
        agent = self._get_agent_for_step(run_data)
        return await agent.execute()

    def _get_agent_for_step(
        self,
        run_data: RunContext,
    ) -> ReActAgent:
        """Get the appropriate agent for executing the step."""
        tools = []
        for tool in self.tools:
            if isinstance(tool, str):
                tool_wrapper = ToolCallWrapper.from_tool_id(
                    tool, run_data.tool_registry, run_data.storage, run_data.plan_run
                )
            else:
                if tool.id not in run_data.tool_registry:
                    run_data.tool_registry.with_tool(tool)
                tool_wrapper = ToolCallWrapper(
                    child_tool=tool,
                    storage=run_data.storage,
                    plan_run=run_data.plan_run,
                )
            if tool_wrapper is not None:
                tools.append(tool_wrapper)
        task = self._template_references(self.task, run_data)
        task_data = self._resolve_input_references_with_descriptions(self.inputs, run_data)

        return ReActAgent(
            task=task,
            task_data=task_data,
            tools=tools,
            run_data=run_data,
            tool_call_limit=self.tool_call_limit,
            allow_agent_clarifications=self.allow_agent_clarifications,
            output_schema=self.output_schema,
            model=self.model,
        )

    @override
    def to_step_data(self, plan: Plan) -> StepData:
        """Convert this SingleToolAgentStep to a Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=",".join(tool if isinstance(tool, str) else tool.id for tool in self.tools),
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )
