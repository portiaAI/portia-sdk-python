"""Implementation of the single tool agent step."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from portia.builder.step_v2 import StepV2

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.config import ExecutionAgentType
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.logger import logger
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Step
from portia.tool import Tool  # noqa: TC001
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent
    from portia.run_context import RunContext


class SingleToolAgentStep(StepV2):
    """A step where an LLM agent intelligently uses a specific tool to complete a task.

    Unlike InvokeToolStep which requires you to specify exact tool arguments, this step
    allows an LLM agent to determine how to use the tool based on the task description
    and available context. The agent will call the tool at most once during execution.
    """

    task: str = Field(description="Natural language description of the task to accomplish.")
    tool: str | Tool = Field(
        description=(
            "ID of the tool the agent should use to complete the task or the Tool instance itself."
        )
    )
    inputs: list[Any] = Field(
        default_factory=list,
        description=(
            "Additional context data for the agent. Can include references to previous step "
            "outputs (using StepOutput), plan inputs (using Input), or literal values. "
            "The agent will use this context to determine how to call the tool."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class defining the expected structure of the agent's output. "
            "If provided, the output from the agent will be coerced to match this schema."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        tool_name = self.tool if isinstance(self.tool, str) else self.tool.id
        return f"SingleToolAgentStep(task='{self.task}', tool='{tool_name}'{output_info})"

    @override
    @traceable(name="Single Tool Agent Step - Run")
    async def run(self, run_data: RunContext) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Run the agent and return its output."""
        agent = self._get_agent_for_step(run_data)
        return await agent.execute_async()

    def _get_agent_for_step(
        self,
        run_data: RunContext,
    ) -> BaseExecutionAgent:
        """Get the appropriate agent for executing the step."""
        if isinstance(self.tool, str):
            tool = ToolCallWrapper.from_tool_id(
                self.tool,
                run_data.tool_registry,
                run_data.storage,
                run_data.plan_run,
            )
        else:
            if self.tool.id not in run_data.tool_registry:
                run_data.tool_registry.with_tool(self.tool)
            tool = ToolCallWrapper(
                child_tool=self.tool,
                storage=run_data.storage,
                plan_run=run_data.plan_run,
            )
        cls: type[BaseExecutionAgent]
        match run_data.config.execution_agent_type:
            case ExecutionAgentType.ONE_SHOT:
                cls = OneShotAgent
            case ExecutionAgentType.DEFAULT:
                cls = DefaultExecutionAgent
        cls = OneShotAgent if isinstance(tool, LLMTool) else cls
        logger().debug(
            f"Using agent: {type(cls).__name__}",
            plan=str(run_data.plan.id),
            plan_run=str(run_data.plan_run.id),
        )
        return cls(
            run_data.legacy_plan,
            run_data.plan_run,
            run_data.config,
            run_data.storage,
            run_data.end_user,
            tool,
            execution_hooks=run_data.execution_hooks,
        )

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this SingleToolAgentStep to a Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=self.tool if isinstance(self.tool, str) else self.tool.id,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )
