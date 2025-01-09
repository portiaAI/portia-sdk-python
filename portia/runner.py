"""Runner classes which actually plan + run queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.agents.base_agent import Output
from portia.agents.one_shot_agent import OneShotAgent
from portia.agents.toolless_agent import ToolLessAgent
from portia.agents.verifier_agent import VerifierAgent
from portia.clarification import (
    Clarification,
)
from portia.config import AgentType, Config, StorageClass
from portia.errors import (
    InvalidAgentOutputError,
    InvalidStorageError,
    InvalidWorkflowStateError,
    PlanError,
)
from portia.llm_wrapper import BaseLLMWrapper, LLMWrapper
from portia.logging import logger, logger_manager
<<<<<<< HEAD
=======
from portia.plan import Output, Plan, ReadOnlyStep, Step
>>>>>>> 935f1de (tests)
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
            case _:
                raise InvalidStorageError(config.storage_class)

    def run_query(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_workflows: list[Plan] | None = None,
    ) -> Workflow:
        """Plan and run a query in one go."""
        plan = self.plan_query(query, tools, example_workflows)
        return self.create_and_execute_workflow(plan)

    def plan_query(
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
            system_context_extension=self.config.planner_system_context_extension,
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

    def create_and_execute_workflow(self, plan: Plan) -> Workflow:
        """Create a new workflow from a plan and then run it."""
        workflow = plan.create_workflow()
        return self._execute_workflow(plan, workflow)

    def execute_workflow(self, workflow: Workflow) -> Workflow:
        """Run a workflow."""
        if workflow.state not in [
            WorkflowState.NOT_STARTED,
            WorkflowState.IN_PROGRESS,
            WorkflowState.NEED_CLARIFICATION,
        ]:
            raise InvalidWorkflowStateError(workflow.id)

        plan = self.storage.get_plan(plan_id=workflow.plan_id)
        return self._execute_workflow(plan, workflow)

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

            # we pass read only copies of the state to the agent so that the runner remains
            # responsible for handling the output of the agent and updating the state.
            agent = self._get_agent_for_step(
                step=ReadOnlyStep.from_step(step),
                workflow=ReadOnlyWorkflow.from_workflow(workflow),
                config=self.config,  # config is already frozen so we don't need to copy
            )

            logger.debug(
                f"Using agent: {type(agent)}",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            try:
                step_output = agent.execute_sync()
                if not isinstance(step_output, Output):
                    raise InvalidAgentOutputError(step_output)  # noqa: TRY301

            except Exception as e:  # noqa: BLE001 - We want to capture all failures here
                error_output = Output(value=str(e))
                workflow.step_outputs[step.output] = error_output
                workflow.state = WorkflowState.FAILED
                workflow.final_output = error_output
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
                workflow.step_outputs[step.output] = step_output
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
        if workflow.final_output:
            logger.info(
                "{output}",
                extra={"plan": plan.id, "workflow": workflow.id},
                output=str(workflow.final_output.value),
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
            case _:
                raise InvalidWorkflowStateError

        return cls(
            step,
            workflow,
            config,
            tool,
        )
