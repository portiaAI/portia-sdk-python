"""Runner classes which actually plan + run queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.agents.one_shot_agent import OneShotAgent
from portia.agents.verifier_agent import VerifierAgent
from portia.clarification import Clarification, InputClarification, MultiChoiceClarification
from portia.config import AgentType, Config, StorageClass
from portia.errors import (
    InvalidAgentError,
    InvalidAgentUsageError,
    InvalidStorageError,
    InvalidWorkflowStateError,
    PlanError,
)
from portia.llm_wrapper import LLMWrapper
from portia.plan import Output, Plan, Step
from portia.planner import Planner
from portia.storage import DiskFileStorage, InMemoryStorage
from portia.tool_registry import LocalToolRegistry, ToolRegistry, ToolSet
from portia.workflow import Workflow, WorkflowState

if TYPE_CHECKING:
    from portia.agents.base_agent import BaseAgent
    from portia.config import Config


class Runner:
    """Create and run plans for queries."""

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize storage and tools."""
        self.config = config
        self.tool_registry = tool_registry or LocalToolRegistry()

        match config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=config.must_get("storage_dir", str))
            case _:
                raise InvalidStorageError(config.storage_class.name)

    def run_query(
        self,
        query: str,
        tools: ToolSet | None = None,
        example_workflows: list[Plan] | None = None,
    ) -> Workflow:
        """Plan and run a query in one go."""
        plan = self.plan_query(query, tools, example_workflows)
        return self.run_plan(plan)

    def plan_query(
        self,
        query: str,
        tools: ToolSet | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples."""
        if not tools:
            tools = self.tool_registry.match_tools(query)

        planner = Planner(config=self.config)
        outcome = planner.generate_plan_or_error(
            query=query,
            tool_list=tools,
            system_context=self.config.planner_system_context_override,
            examples=example_plans,
        )
        if outcome.error:
            raise PlanError(outcome.error)
        self.storage.save_plan(outcome.plan)
        return outcome.plan

    def run_plan(self, plan: Plan) -> Workflow:
        """Run a plan returning the completed workflow or clarifications if needed."""
        workflow = Workflow(plan_id=plan.id, state=WorkflowState.IN_PROGRESS)
        return self._execute_workflow(plan, workflow)

    def resume_workflow(self, workflow: Workflow) -> Workflow:
        """Resume a workflow after an interruption."""
        if workflow.state not in [WorkflowState.IN_PROGRESS, WorkflowState.NEED_CLARIFICATION]:
            raise InvalidWorkflowStateError(workflow.id)
        plan = self.storage.get_plan(plan_id=workflow.plan_id)
        return self._execute_workflow(plan, workflow)

    def get_clarifications_for_step(self, workflow: Workflow) -> list[Clarification]:
        """get_clarifications_for_step isolates clarifications relevant for the current step."""
        return [
            clarification
            for clarification in workflow.clarifications
            if isinstance(clarification, (InputClarification, MultiChoiceClarification))
            and clarification.step == workflow.current_step_index
        ]

    def _execute_workflow(self, plan: Plan, workflow: Workflow) -> Workflow:
        self.storage.save_workflow(workflow)
        for index in range(workflow.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            workflow.current_step_index = index

            agent = self._get_agent_for_step(
                step,
                self.get_clarifications_for_step(workflow),
                self.config.default_agent_type,
            )

            try:
                step_output = agent.execute_sync(
                    llm=LLMWrapper(config=self.config).to_langchain(),
                    step_outputs=workflow.step_outputs,
                )
            except Exception as e:  # noqa: BLE001
                workflow.step_outputs[step.output] = Output(value=str(e))
                workflow.state = WorkflowState.FAILED
                self.storage.save_workflow(workflow)
                return workflow
            else:
                workflow.step_outputs[step.output] = step_output
            self.storage.save_workflow(workflow)

        workflow.state = WorkflowState.COMPLETE
        self.storage.save_workflow(workflow)
        return workflow

    def _get_agent_for_step(
        self,
        step: Step,
        clarifications: list[Clarification],
        agent_type: AgentType,
    ) -> BaseAgent:
        tool = None
        if step.tool_name:
            tool = self.tool_registry.get_tool(step.tool_name)
        match agent_type:
            case AgentType.TOOL_LESS:
                raise NotImplementedError("Toolless agent not implemented in plan executor")
            case AgentType.ONE_SHOT:
                return OneShotAgent(
                    description=step.task,
                    inputs=step.input or [],
                    clarifications=clarifications,
                    tool=tool,
                    system_context=self.config.agent_system_context_override,
                )
            case AgentType.VERIFIER:
                if tool is None:
                    raise InvalidAgentUsageError(agent_type.name)
                return VerifierAgent(
                    description=step.task,
                    inputs=step.input or [],
                    clarifications=clarifications,
                    tool=tool,
                    system_context=self.config.agent_system_context_override,
                )
            case _:
                raise InvalidAgentError(agent_type)
