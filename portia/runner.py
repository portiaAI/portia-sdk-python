"""Runner classes which actually plan + run queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.agents.one_shot_agent import OneShotAgent
from portia.agents.verifier_agent import VerifierAgent
from portia.clarification import (
    Clarification,
    InputClarification,
    MultiChoiceClarification,
)
from portia.config import AgentType, Config, StorageClass
from portia.errors import (
    InvalidStorageError,
    InvalidWorkflowStateError,
    PlanError,
)
from portia.llm_wrapper import LLMWrapper
from portia.logging import logger, logger_manager
from portia.plan import Output, Plan, Step
from portia.planner import Planner
from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
from portia.tool_registry import ToolRegistry, ToolSet
from portia.workflow import Workflow, WorkflowState

if TYPE_CHECKING:
    from portia.agents.base_agent import BaseAgent
    from portia.config import Config


class Runner:
    """Create and run plans for queries."""

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        """Initialize storage and tools."""
        logger_manager.configure_from_config(config)
        self.config = config
        self.tool_registry = tool_registry

        match config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=config.must_get("storage_dir", str))
            case StorageClass.CLOUD:
                self.storage = PortiaCloudStorage(config=config)
            case _:
                raise InvalidStorageError(config.storage_class)

    def run_query(
        self,
        query: str,
        tools: ToolSet | list[str] | None = None,
        example_workflows: list[Plan] | None = None,
    ) -> Workflow:
        """Plan and run a query in one go."""
        plan = self.plan_query(query, tools, example_workflows)
        return self.run_plan(plan)

    def plan_query(
        self,
        query: str,
        tools: ToolSet | list[str] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples."""
        if not tools:
            tools = self.tool_registry.match_tools(query)

        if isinstance(tools, list):
            tools = ToolSet([self.tool_registry.get_tool(tool) for tool in tools])

        planner = Planner(config=self.config)
        logger.debug(f"Running planner for query - {query}")
        outcome = planner.generate_plan_or_error(
            query=query,
            tool_list=tools,
            system_context=self.config.planner_system_context_override,
            examples=example_plans,
        )
        if outcome.error:
            logger.error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)
        self.storage.save_plan(outcome.plan)
        logger.debug(
            f"Plan created with {len(outcome.plan.steps)} steps",
            extra={"plan": outcome.plan.id},
        )
        return outcome.plan

    def run_plan(self, plan: Plan) -> Workflow:
        """Run a plan returning the completed workflow or clarifications if needed."""
        workflow = Workflow(plan_id=plan.id, state=WorkflowState.IN_PROGRESS)
        return self._execute_workflow(plan, workflow)

    def resume_workflow(self, workflow: Workflow) -> Workflow:
        """Resume a workflow after an interruption."""
        if workflow.state not in [
            WorkflowState.IN_PROGRESS,
            WorkflowState.NEED_CLARIFICATION,
        ]:
            raise InvalidWorkflowStateError(workflow.id)
        plan = self.storage.get_plan(plan_id=workflow.plan_id)
        return self._execute_workflow(plan, workflow)

    def get_clarifications_for_step(self, workflow: Workflow, step: int) -> list[Clarification]:
        """get_clarifications_for_step isolates clarifications relevant for the current step."""
        return [
            clarification
            for clarification in workflow.clarifications
            if isinstance(clarification, (InputClarification, MultiChoiceClarification))
            and clarification.step == step
        ]

    def _execute_workflow(self, plan: Plan, workflow: Workflow) -> Workflow:
        self.storage.save_workflow(workflow)
        logger.debug(
            f"Executing workflow from step {workflow.current_step_index}",
            extra={"plan": plan.id, "workflow": workflow.id},
        )
        for index in range(workflow.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            logger.debug(
                f"Executing step {index}: {step.task}",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            workflow.current_step_index = index
            agent = self._get_agent_for_step(
                step,
                self.get_clarifications_for_step(workflow, index),
                self.config.default_agent_type,
            )
            logger.debug(
                f"Using agent: {type(agent)}",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            try:
                step_output = agent.execute_sync(
                    llm=LLMWrapper(config=self.config).to_langchain(),
                    step_outputs=workflow.step_outputs,
                )

            except Exception as e:  # noqa: BLE001 - We want to capture all failures here
                error_output = Output(value=str(e))
                workflow.step_outputs[step.output] = error_output
                workflow.state = WorkflowState.FAILED
                workflow.final_output = error_output
                self.storage.save_workflow(workflow)
                logger.error(
                    f"Step {index} error: {e}",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                logger.debug(
                    f"Final workflow status: {workflow.state}",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                return workflow
            else:
                workflow.step_outputs[step.output] = step_output
                logger.debug(
                    f"Step {index} successful",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )

            # if a clarification was returned append it to the set of clarifications needed
            if isinstance(step_output.value, Clarification) or (
                isinstance(step_output.value, list)
                and len(step_output.value) > 0
                and all(isinstance(item, Clarification) for item in step_output.value)
            ):
                new_clarifications = (
                    [step_output.value]
                    if isinstance(step_output.value, Clarification)
                    else step_output.value
                )
                for clarification in new_clarifications:
                    clarification.step = workflow.current_step_index

                workflow.clarifications = workflow.clarifications + new_clarifications
                workflow.state = WorkflowState.NEED_CLARIFICATION
                self.storage.save_workflow(workflow)
                logger.info(
                    f"{len(new_clarifications)} Clarification(s) requested",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                return workflow

            # set final output if is last step (accounting for zero index)
            if index == len(plan.steps) - 1:
                workflow.final_output = step_output

            # persist at the end of each step
            self.storage.save_workflow(workflow)

        workflow.state = WorkflowState.COMPLETE
        self.storage.save_workflow(workflow)
        logger.debug(
            f"Final workflow status: {workflow.state}",
            extra={"plan": plan.id, "workflow": workflow.id},
        )
        if workflow.final_output:
            logger.info(
                str(workflow.final_output.value),
                extra={"plan": plan.id, "workflow": workflow.id},
            )
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
                return VerifierAgent(
                    description=step.task,
                    inputs=step.input or [],
                    clarifications=clarifications,
                    tool=tool,
                    system_context=self.config.agent_system_context_override,
                )
            case _:
                raise InvalidWorkflowStateError
