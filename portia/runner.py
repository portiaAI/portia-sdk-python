"""Runner classes that plan and execute workflows for queries.

This module contains the core classes responsible for generating, managing, and executing workflows
in response to queries. The `Runner` class serves as the main entry point, orchestrating the
planning and execution process. It uses various agents and tools to carry out tasks step by step,
saving the state of the workflow at each stage. It also handles error cases, clarification
requests, and workflow state transitions.

The `Runner` class provides methods to:

- Generate a plan for executing a query.
- Create and manage workflows.
- Execute workflows step by step, using agents to handle the execution of tasks.
- Resolve clarifications required during the execution of workflows.
- Wait for workflows to reach a state where they can be resumed.

Modules in this file work with different storage backends (memory, disk, cloud) and can handle
complex queries using various planner and agent configurations.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from portia.agents.base_agent import Output
from portia.agents.one_shot_agent import OneShotAgent
from portia.agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.agents.verifier_agent import VerifierAgent
from portia.clarification import (
    Clarification,
)
from portia.config import AgentType, Config, PlannerType, StorageClass
from portia.errors import (
    InvalidWorkflowStateError,
    PlanError,
)
from portia.execution_context import (
    execution_context,
    get_execution_context,
    is_execution_context_set,
)
from portia.logger import logger, logger_manager
from portia.plan import Plan, ReadOnlyPlan, ReadOnlyStep, Step
from portia.planners.one_shot_planner import OneShotPlanner
from portia.storage import (
    DiskFileStorage,
    InMemoryStorage,
    PortiaCloudStorage,
)
from portia.tool import ToolRunContext
from portia.tool_registry import InMemoryToolRegistry, ToolRegistry
from portia.tool_wrapper import ToolCallWrapper
from portia.workflow import ReadOnlyWorkflow, Workflow, WorkflowState, WorkflowUUID

if TYPE_CHECKING:
    from portia.agents.base_agent import BaseAgent
    from portia.config import Config
    from portia.plan import Plan
    from portia.planners.planner import Planner
    from portia.tool import Tool


class Runner:
    """Runner class is the top level abstraction and entrypoint for most programs using the SDK.

    The runner is responsible for intermediating planning via Planners and execution via Agents.
    """

    def __init__(
        self,
        config: Config,
        tools: ToolRegistry | list[Tool],
    ) -> None:
        """Initialize storage and tools.

        Args:
            config (Config): The configuration to initialize the runner.
            tools (ToolRegistry | list[Tool]): The registry or list of tools to use.

        """
        logger_manager.configure_from_config(config)
        self.config = config
        self.tool_registry = (
            InMemoryToolRegistry.from_local_tools(tools) if isinstance(tools, list) else tools
        )

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
        """End-to-end function to generate a plan and then execute it.

        This is the simplest way to plan and execute a query using the SDK.

        Args:
            query (str): The query to be executed.
            tools (list[Tool] | list[str] | None): List of tools to use for the query.
            If not provided all tools in the registry will be used.
            example_plans (list[Plan] | None): Optional list of example plans. If not
            provide a default set of example plans will be used.

        Returns:
            Workflow: The workflow resulting from executing the query.

        """
        plan = self.generate_plan(query, tools, example_plans)
        workflow = self.create_workflow(plan)
        return self.execute_workflow(workflow)

    def generate_plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples.

        Args:
            query (str): The query to generate the plan for.
            tools (list[Tool] | list[str] | None): List of tools to use for the query.
            If not provided all tools in the registry will be used.
            example_plans (list[Plan] | None): Optional list of example plans. If not
            provide a default set of example plans will be used.

        Returns:
            Plan: The plan for executing the query.

        Raises:
            PlanError: If there is an error while generating the plan.

        """
        if isinstance(tools, list):
            tools = [
                self.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.tool_registry.match_tools(query)

        logger().debug(f"Running planner for query - {query}")
        planner = self._get_planner()
        outcome = planner.generate_plan_or_error(
            ctx=get_execution_context(),
            query=query,
            tool_list=tools,
            examples=example_plans,
        )
        if outcome.error:
            logger().error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)
        self.storage.save_plan(outcome.plan)
        logger().info(
            f"Plan created with {len(outcome.plan.steps)} steps",
            extra={"plan": outcome.plan.id},
        )
        logger().debug(
            "Plan: {plan}",
            extra={"plan": outcome.plan.id},
            plan=outcome.plan.model_dump_json(indent=4),
        )

        return outcome.plan

    def create_workflow(self, plan: Plan) -> Workflow:
        """Create a workflow from a Plan.

        Args:
            plan (Plan): The plan to create a workflow from.

        Returns:
            Workflow: The created workflow.

        """
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
        workflow_id: WorkflowUUID | str | None = None,
    ) -> Workflow:
        """Run a workflow.

        Args:
            workflow (Workflow | None): The workflow to execute. Defaults to None.
            workflow_id (WorkflowUUID | str | None): The ID of the workflow to execute. Defaults to
                None.

        Returns:
            Workflow: The resulting workflow after execution.

        Raises:
            ValueError: If neither workflow nor workflow_id is provided.
            InvalidWorkflowStateError: If the workflow is not in a valid state to be executed.

        """
        if not workflow:
            if not workflow_id:
                raise ValueError("Either workflow or workflow_id must be provided")

            parsed_id = (
                WorkflowUUID.from_string(workflow_id)
                if isinstance(workflow_id, str)
                else workflow_id
            )
            workflow = self.storage.get_workflow(parsed_id)

        if workflow.state not in [
            WorkflowState.NOT_STARTED,
            WorkflowState.IN_PROGRESS,
            WorkflowState.NEED_CLARIFICATION,
            WorkflowState.READY_TO_RESUME,
        ]:
            raise InvalidWorkflowStateError(workflow.id)

        plan = self.storage.get_plan(plan_id=workflow.plan_id)

        # if the workflow has execution context associated, but none is set then use it
        if not is_execution_context_set():
            with execution_context(workflow.execution_context):
                return self._execute_workflow(plan, workflow)

        # if there is execution context set, make sure we update the workflow before running
        workflow.execution_context = get_execution_context()
        return self._execute_workflow(plan, workflow)

    def resolve_clarification(
        self,
        workflow: Workflow,
        clarification: Clarification,
        response: object,
    ) -> Workflow:
        """Resolve a clarification updating the workflow state as needed.

        Args:
            workflow (Workflow): The workflow being updated.
            clarification (Clarification): The clarification to resolve.
            response (object): The response to the clarification.

        Returns:
            Workflow: The updated workflow.

        """
        clarification.resolved = True
        clarification.response = response
        if len(workflow.get_outstanding_clarifications()) == 0:
            workflow.state = WorkflowState.READY_TO_RESUME
        self.storage.save_workflow(workflow)
        return workflow

    def wait_for_ready(
        self,
        workflow: Workflow,
        max_retries: int = 6,
        backoff_start_time_seconds: int = 7 * 60,
        backoff_time_seconds: int = 2,
    ) -> Workflow:
        """Wait for the workflow to be in a state that it can be re-run.

        This is generally because there are outstanding clarifications that need to be resolved.

        Args:
            workflow (Workflow): The workflow to wait for.
            max_retries (int): The maximum number of retries to wait for the workflow to be ready
                after the backoff period starts.
            backoff_start_time_seconds (int): The time after which the backoff period starts.
            backoff_time_seconds (int): The time to wait between retries after the backoff period
                starts.

        Returns:
            Workflow: The updated workflow once it is ready to be re-run.

        Raises:
            InvalidWorkflowStateError: If the workflow cannot be waited for.

        """
        start_time = time.time()
        tries = 0
        if workflow.state not in [
            WorkflowState.IN_PROGRESS,
            WorkflowState.NOT_STARTED,
            WorkflowState.READY_TO_RESUME,
            WorkflowState.NEED_CLARIFICATION,
        ]:
            raise InvalidWorkflowStateError("Cannot wait for workflow that is not ready to run")

        # These states can continue straight away
        if workflow.state in [
            WorkflowState.IN_PROGRESS,
            WorkflowState.NOT_STARTED,
            WorkflowState.READY_TO_RESUME,
        ]:
            return workflow

        plan = self.storage.get_plan(workflow.plan_id)
        while workflow.state != WorkflowState.READY_TO_RESUME:
            if tries >= max_retries:
                raise InvalidWorkflowStateError("Workflow is not ready to resume after max retries")

            # if we've waited longer than the backoff time, start the backoff period
            if time.time() - start_time > backoff_start_time_seconds:
                tries += 1
                backoff_time_seconds *= 2

            # wait a couple of seconds as we're long polling
            time.sleep(backoff_time_seconds)

            # refresh state
            workflow = self.storage.get_workflow(workflow.id)

            # if its not ready we can see if the tool is ready
            if workflow.state != WorkflowState.READY_TO_RESUME:
                step = plan.steps[workflow.current_step_index]
                next_tool = self._get_tool_for_step(step, workflow)
                if next_tool:
                    tool_ready = next_tool.ready(
                        ToolRunContext(
                            execution_context=workflow.execution_context,
                            workflow_id=workflow.id,
                            config=self.config,
                            clarifications=workflow.get_clarifications_for_step(),
                        ),
                    )
                    logger().debug(f"Tool state for {next_tool.name} is ready={tool_ready}")
                    if tool_ready:
                        workflow.state = WorkflowState.READY_TO_RESUME
                        self.storage.save_workflow(workflow)

            logger().debug(f"New workflow state for {workflow.id} is {workflow.state}")

        logger().info(f"Workflow {workflow.id} is ready to resume")

        return workflow

    def _execute_workflow(self, plan: Plan, workflow: Workflow) -> Workflow:
        """Execute the workflow steps, updating the workflow state as needed.

        Args:
            plan (Plan): The plan to execute.
            workflow (Workflow): The workflow to execute.

        Returns:
            Workflow: The updated workflow after execution.

        """
        workflow.state = WorkflowState.IN_PROGRESS
        self.storage.save_workflow(workflow)
        logger().debug(
            f"Executing workflow from step {workflow.current_step_index}",
            extra={"plan": plan.id, "workflow": workflow.id},
        )
        for index in range(workflow.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            workflow.current_step_index = index
            logger().debug(
                f"Executing step {index}: {step.task}",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            # we pass read only copies of the state to the agent so that the runner remains
            # responsible for handling the output of the agent and updating the state.
            agent = self._get_agent_for_step(
                step=ReadOnlyStep.from_step(step),
                workflow=ReadOnlyWorkflow.from_workflow(workflow),
            )
            logger().debug(
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
                logger().error(
                    "error: {error}",
                    error=e,
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                logger().debug(
                    f"Final workflow status: {workflow.state}",
                    extra={"plan": plan.id, "workflow": workflow.id},
                )
                return workflow
            else:
                workflow.outputs.step_outputs[step.output] = step_output
                logger().debug(
                    "Step output - {output}",
                    extra={"plan": plan.id, "workflow": workflow.id},
                    output=str(step_output.value),
                )

            if self._handle_clarifications(workflow, step_output, plan):
                return workflow

            # set final output if is last step (accounting for zero index)
            if index == len(plan.steps) - 1:
                workflow.outputs.final_output = self._get_final_output(plan, workflow, step_output)

            # persist at the end of each step
            self.storage.save_workflow(workflow)
            logger().debug(
                "New Workflow State: {workflow}",
                extra={"plan": plan.id, "workflow": workflow.id},
                workflow=workflow.model_dump_json(indent=4),
            )

        workflow.state = WorkflowState.COMPLETE
        self.storage.save_workflow(workflow)
        logger().debug(
            f"Final workflow status: {workflow.state}",
            extra={"plan": plan.id, "workflow": workflow.id},
        )
        if workflow.outputs.final_output:
            logger().info(
                "{output}",
                extra={"plan": plan.id, "workflow": workflow.id},
                output=str(workflow.outputs.final_output.value),
            )
        return workflow

    def _get_planner(self) -> Planner:
        """Get the planner based on the configuration.

        Returns:
            Planner: The planner to be used for generating plans.

        """
        cls: type[Planner]
        match self.config.default_planner:
            case PlannerType.ONE_SHOT:
                cls = OneShotPlanner

        return cls(self.config)

    def _get_final_output(self, plan: Plan, workflow: Workflow, step_output: Output) -> Output:
        """Get the final output and add summarization to it.

        Args:
            plan (Plan): The plan to execute.
            workflow (Workflow): The workflow to execute.
            step_output (Output): The output of the last step.

        """
        final_output = Output(
            value=step_output.value,
            summary=None,
        )

        try:
            summarizer = FinalOutputSummarizer(config=self.config)
            summary = summarizer.create_summary(
                workflow=ReadOnlyWorkflow.from_workflow(workflow),
                plan=ReadOnlyPlan.from_plan(plan),
            )
            final_output.summary = summary

        except Exception as e:  # noqa: BLE001
            logger().warning(f"Error summarising workflow: {e}")

        return final_output

    def _handle_clarifications(self, workflow: Workflow, step_output: Output, plan: Plan) -> bool:
        """Handle any clarifications needed during workflow execution.

        Args:
            workflow (Workflow): The workflow to execute.
            step_output (Output): The output of the last step.
            plan (Plan): The plan to execute.

        Returns:
            bool: True if clarification is needed and workflow execution should stop.

        """
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

            workflow.outputs.clarifications = workflow.outputs.clarifications + new_clarifications
            workflow.state = WorkflowState.NEED_CLARIFICATION
            self.storage.save_workflow(workflow)
            logger().info(
                f"{len(new_clarifications)} Clarification(s) requested",
                extra={"plan": plan.id, "workflow": workflow.id},
            )
            return True
        return False

    def _get_tool_for_step(self, step: Step, workflow: Workflow) -> Tool | None:
        if not step.tool_id:
            return None
        child_tool = self.tool_registry.get_tool(step.tool_id)
        return ToolCallWrapper(
            child_tool=child_tool,
            storage=self.storage,
            workflow=workflow,
        )

    def _get_agent_for_step(
        self,
        step: Step,
        workflow: Workflow,
    ) -> BaseAgent:
        """Get the appropriate agent for executing a given step.

        Args:
            step (Step): The step for which the agent is needed.
            workflow (Workflow): The workflow associated with the step.

        Returns:
            BaseAgent: The agent to execute the step.

        """
        tool = self._get_tool_for_step(step, workflow)
        cls: type[BaseAgent]
        match self.config.default_agent_type:
            case AgentType.ONE_SHOT:
                cls = OneShotAgent
            case AgentType.VERIFIER:
                cls = VerifierAgent
        return cls(
            step,
            workflow,
            self.config,
            tool,
        )
