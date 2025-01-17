"""Runner classes which actually plan + run queries."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from portia.agents.base_agent import Output
from portia.agents.one_shot_agent import OneShotAgent
from portia.agents.toolless_agent import ToolLessAgent
from portia.agents.verifier_agent import VerifierAgent
from portia.clarification import (
    Clarification,
)
from portia.config import AgentType, Config, StorageClass
from portia.context import execution_context, get_execution_context, is_execution_context_set
from portia.errors import (
    InvalidWorkflowStateError,
    PlanError,
)
from portia.llm_wrapper import BaseLLMWrapper, LLMWrapper
from portia.logger import logger, logger_manager
from portia.plan import Plan, ReadOnlyStep, Step
from portia.planner import Planner
from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
from portia.workflow import ReadOnlyWorkflow, Workflow, WorkflowState

if TYPE_CHECKING:
    from portia.agents.base_agent import BaseAgent
    from portia.config import Config
    from portia.plan import Plan, Step
    from portia.tool import Tool
    from portia.tool_registry import ToolRegistry


class Runner:
    """Create and run plans for queries."""

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry,
        llm_wrapper_class: type[BaseLLMWrapper] | None = None,
    ) -> None:
        """Initialize storage and tools."""
        logger_manager.configure_from_config(config)
        self.config = config
        self.tool_registry = tool_registry
        self.llm_wrapper_class = llm_wrapper_class or LLMWrapper

        match config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=config.must_get("storage_dir", str))
            case StorageClass.CLOUD:
                self.storage = PortiaCloudStorage(config=config)

    def execute_query(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Workflow:
        """End to end function to generate a plan and then execute it."""
        plan = self.generate_plan(query, tools, example_plans)
        workflow = self.create_workflow(plan)
        return self.execute_workflow(workflow)

    def generate_plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples."""
        if isinstance(tools, list):
            tools = [
                self.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.tool_registry.match_tools(query)

        logger.debug(f"Running planner for query - {query}")
        planner = Planner(self.llm_wrapper_class(self.config))
        outcome = planner.generate_plan_or_error(
            query=query,
            tool_list=tools,
            examples=example_plans,
        )
        if outcome.error:
            logger.error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)
        self.storage.save_plan(outcome.plan)
        logger.info(
            f"Plan created with {len(outcome.plan.steps)} steps",
            extra={"plan": outcome.plan.id},
        )
        logger.debug(
            "Plan: {plan}",
            extra={"plan": outcome.plan.id},
            plan=outcome.plan.model_dump_json(indent=4),
        )

        return outcome.plan

    def create_workflow(self, plan: Plan) -> Workflow:
        """Create a workflow from a Plan."""
        workflow = Workflow(
            plan_id=plan.id,
            state=WorkflowState.NOT_STARTED,
            execution_context=get_execution_context(),
        )
        self.storage.save_workflow(workflow)
        return workflow

    def execute_workflow(
        self,
        workflow: Workflow | None = None,
        workflow_id: UUID | str | None = None,
    ) -> Workflow:
        """Run a workflow."""
        if not workflow:
            if not workflow_id:
                raise ValueError("Either workflow or workflow_id must be provided")

            parsed_id = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id
            workflow = self.storage.get_workflow(parsed_id)

        if workflow.state not in [
            WorkflowState.NOT_STARTED,
            WorkflowState.IN_PROGRESS,
            WorkflowState.NEED_CLARIFICATION,
        ]:
            raise InvalidWorkflowStateError(workflow.id)

        plan = self.storage.get_plan(plan_id=workflow.plan_id)

        # if the workflow has execution context associated, but none is set then use it
        if not is_execution_context_set() and workflow.execution_context:
            with execution_context(workflow.execution_context):
                return self._execute_workflow(plan, workflow)

        # if there is execution context set, make sure we update the workflow before running
        if is_execution_context_set():
            workflow.execution_context = get_execution_context()

        return self._execute_workflow(plan, workflow)

    def _execute_workflow(self, plan: Plan, workflow: Workflow) -> Workflow:
        workflow.state = WorkflowState.IN_PROGRESS
        self.storage.save_workflow(workflow)
        logger.debug(
            f"Executing workflow from step {workflow.current_step_index}",
            extra={"plan": plan.id, "workflow": workflow.id},
        )
        for index in range(workflow.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            workflow.current_step_index = index
            logger.debug(
                f"Executing step {index}: {step.task}",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            # we pass read only copies of the state to the agent so that the runner remains
            # responsible for handling the output of the agent and updating the state.
            agent = self._get_agent_for_step(
                step=ReadOnlyStep.from_step(step),
                workflow=ReadOnlyWorkflow.from_workflow(workflow),
                config=self.config,
            )
            logger.debug(
                f"Using agent: {type(agent)}",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            try:
                step_output = agent.execute_sync()
            except Exception as e:  # noqa: BLE001 - We want to capture all failures here
                error_output = Output(value=str(e))
                workflow.outputs.step_outputs[step.output] = error_output
                workflow.state = WorkflowState.FAILED
                workflow.outputs.final_output = error_output
                self.storage.save_workflow(workflow)
                logger.error(
                    "error: {error}",
                    error=e,
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                logger.debug(
                    f"Final workflow status: {workflow.state}",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                return workflow
            else:
                workflow.outputs.step_outputs[step.output] = step_output
                logger.debug(
                    "Step output - {output}",
                    extra={"plan": plan.id, "workflow": workflow.id},
                    output=str(step_output.value),
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

                workflow.outputs.clarifications = (
                    workflow.outputs.clarifications + new_clarifications
                )
                workflow.state = WorkflowState.NEED_CLARIFICATION
                self.storage.save_workflow(workflow)
                logger.info(
                    f"{len(new_clarifications)} Clarification(s) requested",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                return workflow

            # set final output if is last step (accounting for zero index)
            if index == len(plan.steps) - 1:
                workflow.outputs.final_output = step_output

            # persist at the end of each step
            self.storage.save_workflow(workflow)
            logger.debug(
                "New Workflow State: {workflow}",
                extra={"plan": plan.id, "workflow": workflow.id},
                workflow=workflow.model_dump_json(indent=4),
            )

        workflow.state = WorkflowState.COMPLETE
        self.storage.save_workflow(workflow)
        logger.debug(
            f"Final workflow status: {workflow.state}",
            extra={"plan": plan.id, "workflow": workflow.id},
        )
        if workflow.outputs.final_output:
            logger.info(
                "{output}",
                extra={"plan": plan.id, "workflow": workflow.id},
                output=str(workflow.outputs.final_output.value),
            )
        return workflow

    def _get_agent_for_step(
        self,
        step: Step,
        workflow: Workflow,
        config: Config,
    ) -> BaseAgent:
        tool = None
        if step.tool_name:
            tool = self.tool_registry.get_tool(step.tool_name)
        cls: type[BaseAgent]
        match config.default_agent_type:
            case AgentType.TOOL_LESS:
                cls = ToolLessAgent
            case AgentType.ONE_SHOT:
                cls = OneShotAgent
            case AgentType.VERIFIER:
                cls = VerifierAgent

        return cls(
            step,
            workflow,
            config,
            tool,
        )
