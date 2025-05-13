"""Portia classes that plan and execute runs for queries.

This module contains the core classes responsible for generating, managing, and executing plans
in response to queries. The `Portia` class serves as the main entry point, orchestrating the
planning and execution process. It uses various agents and tools to carry out tasks step by step,
saving the state of the run at each stage. It also handles error cases, clarification
requests, and run state transitions.

The `Portia` class provides methods to:

- Generate a plan for executing a query.
- Create and manage runs.
- Execute runs step by step, using agents to handle the execution of tasks.
- Resolve clarifications required during the execution of runs.
- Wait for runs to reach a state where they can be resumed.

Modules in this file work with different storage backends (memory, disk, cloud) and can handle
complex queries using various planning and execution agent configurations.

"""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import TYPE_CHECKING
from uuid import UUID

from portia.clarification import (
    Clarification,
    ClarificationCategory,
)
from portia.cloud import PortiaCloudClient
from portia.config import (
    Config,
    ExecutionAgentType,
    GenerativeModelsConfig,
    PlanningAgentType,
    StorageClass,
)
from portia.end_user import EndUser
from portia.errors import (
    InvalidPlanRunStateError,
    PlanError,
    PlanNotFoundError,
    StorageError,
)
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.one_shot_agent import OneShotAgent
from portia.execution_agents.output import (
    LocalDataValue,
    Output,
)
from portia.execution_agents.utils.final_output_summarizer import FinalOutputSummarizer
from portia.execution_hooks import ExecutionHooks
from portia.introspection_agents.default_introspection_agent import DefaultIntrospectionAgent
from portia.introspection_agents.introspection_agent import (
    COMPLETED_OUTPUT,
    SKIPPED_OUTPUT,
    BaseIntrospectionAgent,
    PreStepIntrospection,
    PreStepIntrospectionOutcome,
)
from portia.logger import logger, logger_manager
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, PlanInput, ReadOnlyPlan, ReadOnlyStep, Step
from portia.plan_run import PlanRun, PlanRunState, PlanRunUUID, ReadOnlyPlanRun
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from portia.prefixed_uuid import PlanUUID
from portia.storage import (
    DiskFileStorage,
    InMemoryStorage,
    PortiaCloudStorage,
)
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import (
    DefaultToolRegistry,
    PortiaToolRegistry,
    ToolRegistry,
)
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.common import Serializable
    from portia.execution_agents.base_execution_agent import BaseExecutionAgent
    from portia.planning_agents.base_planning_agent import BasePlanningAgent


class Portia:
    """Portia client is the top level abstraction and entrypoint for most programs using the SDK.

    It is responsible for intermediating planning via PlanningAgents and
    execution via ExecutionAgents.
    """

    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
        execution_hooks: ExecutionHooks | None = None,
    ) -> None:
        """Initialize storage and tools.

        Args:
            config (Config): The configuration to initialize the Portia client. If not provided, the
                default configuration will be used.
            tools (ToolRegistry | list[Tool]): The registry or list of tools to use. If not
                provided, the open source tool registry will be used, alongside the default tools
                from Portia cloud if a Portia API key is set.
            execution_hooks (ExecutionHooks | None): Hooks that can be used to modify or add
                extra functionality to the run of a plan.

        """
        self.config = config if config else Config.from_default()
        logger_manager.configure_from_config(self.config)
        logger().info(f"Starting Portia v{version('portia-sdk-python')}")
        if self.config.portia_api_key and self.config.portia_api_endpoint:
            logger().info(f"Using Portia cloud API endpoint: {self.config.portia_api_endpoint}")
        self._log_models(self.config)
        self.execution_hooks = execution_hooks if execution_hooks else ExecutionHooks()
        if not self.config.has_api_key("portia_api_key"):
            logger().warning(
                "No Portia API key found, Portia cloud tools and storage will not be available.",
            )

        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, list):
            self.tool_registry = ToolRegistry(tools)
        else:
            self.tool_registry = DefaultToolRegistry(self.config)

        match self.config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=self.config.must_get("storage_dir", str))
            case StorageClass.CLOUD:
                self.storage = PortiaCloudStorage(config=self.config)

    def initialize_end_user(self, end_user: str | EndUser | None = None) -> EndUser:
        """Handle initializing the end_user based on the provided type."""
        default_external_id = "portia:default_user"
        if isinstance(end_user, str):
            if end_user == "":
                end_user = default_external_id
            end_user_instance = self.storage.get_end_user(external_id=end_user)
            if end_user_instance:
                return end_user_instance
            end_user_instance = EndUser(external_id=end_user or default_external_id)
            return self.storage.save_end_user(end_user_instance)

        if not end_user:
            end_user = EndUser(external_id=default_external_id)
            return self.storage.save_end_user(end_user)

        return self.storage.save_end_user(end_user)

    def run(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
    ) -> PlanRun:
        """End-to-end function to generate a plan and then execute it.

        This is the simplest way to plan and execute a query using the SDK.

        Args:
            query (str): The query to be executed.
            tools (list[Tool] | list[str] | None): List of tools to use for the query.
            If not provided all tools in the registry will be used.
            example_plans (list[Plan] | None): Optional list of example plans. If not
            provide a default set of example plans will be used.
            end_user (str | EndUser | None = None): The end user for this plan run.
            plan_run_inputs (list[PlanInput] | list[dict[str, str]] | dict[str, str] | None):
                Provides input values for the run. This can be a list of PlanInput objects, a list
                of dicts with keys "name", "description" (optional) and "value", or a dict of
                plan run input name to value.

        Returns:
            PlanRun: The run resulting from executing the query.

        """
        coerced_plan_run_inputs = self._coerce_plan_run_inputs(plan_run_inputs)
        plan = self.plan(query, tools, example_plans, end_user, coerced_plan_run_inputs)
        end_user = self.initialize_end_user(end_user)
        plan_run = self.create_plan_run(plan, end_user, coerced_plan_run_inputs)
        return self.resume(plan_run)

    def _coerce_plan_run_inputs(
        self,
        plan_run_inputs: list[PlanInput]
        | list[dict[str, Serializable]]
        | dict[str, Serializable]
        | None,
    ) -> list[PlanInput] | None:
        """Coerce plan inputs from any input type into a list of PlanInputs we use internally."""
        if plan_run_inputs is None:
            return None
        if isinstance(plan_run_inputs, list):
            to_return = []
            for plan_run_input in plan_run_inputs:
                if isinstance(plan_run_input, dict):
                    if "name" not in plan_run_input or "value" not in plan_run_input:
                        raise ValueError("Plan input must have a name and value")
                    to_return.append(
                        PlanInput(
                            name=plan_run_input["name"],
                            description=plan_run_input.get("description", None),
                            value=plan_run_input["value"],
                        )
                    )
                else:
                    to_return.append(plan_run_input)
            return to_return
        if isinstance(plan_run_inputs, dict):
            to_return = []
            for key, value in plan_run_inputs.items():
                to_return.append(PlanInput(name=key, value=value))
            return to_return
        raise ValueError("Invalid plan run inputs received")

    def plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: list[Plan] | None = None,
        end_user: str | EndUser | None = None,
        plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples.

        Args:
            query (str): The query to generate the plan for.
            tools (list[Tool] | list[str] | None): List of tools to use for the query.
            If not provided all tools in the registry will be used.
            example_plans (list[Plan] | None): Optional list of example plans. If not
            provide a default set of example plans will be used.
            end_user (str | EndUser | None = None): The optional end user for this plan.
            plan_inputs (list[PlanInput] | None): Optional list of inputs required for the plan.
              This can be a list of Planinput objects, a list of dicts with keys "name" and
              "description" (optional), or a list of plan run input names. If a value is provided
              with a PlanInput object or in a dictionary, it will be ignored as values are only
              used when running the plan.

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

        end_user = self.initialize_end_user(end_user)
        logger().info(f"Running planning_agent for query - {query}")
        planning_agent = self._get_planning_agent()
        coerced_plan_inputs = self._coerce_plan_inputs(plan_inputs)
        outcome = planning_agent.generate_steps_or_error(
            query=query,
            tool_list=tools,
            end_user=end_user,
            examples=example_plans,
            plan_inputs=coerced_plan_inputs,
        )
        if outcome.error:
            if (
                isinstance(self.tool_registry, DefaultToolRegistry)
                and not self.config.portia_api_key
            ):
                self._log_replan_with_portia_cloud_tools(
                    outcome.error,
                    query,
                    end_user,
                    example_plans,
                )
            logger().error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)
        plan = Plan(
            plan_context=PlanContext(
                query=query,
                tool_ids=[tool.id for tool in tools],
            ),
            steps=outcome.steps,
            plan_inputs=coerced_plan_inputs or [],
        )
        self.storage.save_plan(plan)
        logger().info(
            f"Plan created with {len(plan.steps)} steps",
            plan=str(plan.id),
        )
        logger().debug(plan.pretty_print())

        return plan

    def _coerce_plan_inputs(
        self, plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None
    ) -> list[PlanInput] | None:
        """Coerce plan inputs from any input type into a list of PlanInputs we use internally."""
        if plan_inputs is None:
            return None
        if isinstance(plan_inputs, list):
            to_return = []
            for plan_input in plan_inputs:
                if isinstance(plan_input, dict):
                    if "name" not in plan_input:
                        raise ValueError("Plan input must have a name and description")
                    to_return.append(
                        PlanInput(
                            name=plan_input["name"],
                            description=plan_input.get("description", None),
                        )
                    )
                elif isinstance(plan_input, str):
                    to_return.append(PlanInput(name=plan_input))
                else:
                    to_return.append(plan_input)
            return to_return
        raise ValueError("Invalid plan inputs received")

    def run_plan(
        self,
        plan: Plan | PlanUUID | UUID,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput]
        | list[dict[str, Serializable]]
        | dict[str, Serializable]
        | None = None,
    ) -> PlanRun:
        """Run a plan.

        Args:
            plan (Plan | PlanUUID | UUID): The plan to run, or the ID of the plan to load from
              storage.
            end_user (str | EndUser | None = None): The end user to use.
            plan_run_inputs (list[PlanInput] | list[dict[str, Serializable]] | dict[str, Serializable] | None):
              Provides input values for the run. This can be a list of PlanInput objects, a list
              of dicts with keys "name", "description" (optional) and "value", or a dict of
              plan run input name to value.

        Returns:
            PlanRun: The resulting PlanRun object.

        """  # noqa: E501
        # ensure we have the plan in storage.
        # we won't if for example the user used PlanBuilder instead of dynamic planning.
        plan_id = (
            plan
            if isinstance(plan, PlanUUID)
            else PlanUUID(uuid=plan)
            if isinstance(plan, UUID)
            else plan.id
        )
        try:
            plan = self.storage.get_plan(plan_id)
        except (PlanNotFoundError, StorageError):
            if isinstance(plan, Plan):
                self.storage.save_plan(plan)
            else:
                raise PlanNotFoundError(plan_id) from None

        end_user = self.initialize_end_user(end_user)
        coerced_plan_run_inputs = self._coerce_plan_run_inputs(plan_run_inputs)
        plan_run = self.create_plan_run(plan, end_user, coerced_plan_run_inputs)
        return self.resume(plan_run)

    def resume(
        self,
        plan_run: PlanRun | None = None,
        plan_run_id: PlanRunUUID | str | None = None,
    ) -> PlanRun:
        """Resume a PlanRun.

        If a clarification handler was provided as part of the execution hooks, it will be used
        to handle any clarifications that are raised during the execution of the plan run.
        If no clarification handler was provided and a clarification is raised, the run will be
        returned in the `NEED_CLARIFICATION` state. The clarification will then need to be handled
        by the caller before the plan run is resumed.

        Args:
            plan_run (PlanRun | None): The PlanRun to resume. Defaults to None.
            plan_run_id (RunUUID | str | None): The ID of the PlanRun to resume. Defaults to
                None.

        Returns:
            PlanRun: The resulting PlanRun after execution.

        Raises:
            ValueError: If neither plan_run nor plan_run_id is provided.
            InvalidPlanRunStateError: If the plan run is not in a valid state to be resumed.

        """
        if not plan_run:
            if not plan_run_id:
                raise ValueError("Either plan_run or plan_run_id must be provided")

            parsed_id = (
                PlanRunUUID.from_string(plan_run_id)
                if isinstance(plan_run_id, str)
                else plan_run_id
            )
            plan_run = self.storage.get_plan_run(parsed_id)

        if plan_run.state not in [
            PlanRunState.NOT_STARTED,
            PlanRunState.IN_PROGRESS,
            PlanRunState.NEED_CLARIFICATION,
            PlanRunState.READY_TO_RESUME,
        ]:
            raise InvalidPlanRunStateError(plan_run.id)

        plan = self.storage.get_plan(plan_id=plan_run.plan_id)

        return self.execute_plan_run_and_handle_clarifications(plan, plan_run)

    def _process_plan_input_values(
        self,
        plan: Plan,
        plan_run: PlanRun,
        plan_run_inputs: list[PlanInput] | None = None,
    ) -> None:
        """Process plan input values and add them to the plan run.

        Args:
            plan (Plan): The plan containing required inputs.
            plan_run (PlanRun): The plan run to update with input values.
            plan_run_inputs (list[PlanInput] | None): Values for plan inputs.

        Raises:
            ValueError: If required plan inputs are missing.

        """
        if plan.plan_inputs and not plan_run_inputs:
            raise ValueError("Inputs are required for this plan but have not been specified")
        if plan_run_inputs and not plan.plan_inputs:
            logger().warning(
                "Inputs are not required for this plan but plan inputs were provided",
            )

        if plan_run_inputs and plan.plan_inputs:
            input_values_by_name = {input_obj.name: input_obj for input_obj in plan_run_inputs}

            # Validate all required inputs are provided
            missing_inputs = [
                input_obj.name
                for input_obj in plan.plan_inputs
                if input_obj.name not in input_values_by_name
            ]
            if missing_inputs:
                raise ValueError(f"Missing required plan input values: {', '.join(missing_inputs)}")

            for plan_input in plan.plan_inputs:
                if plan_input.name in input_values_by_name:
                    plan_run.plan_run_inputs[plan_input.name] = LocalDataValue(
                        value=input_values_by_name[plan_input.name].value
                    )

            # Check for unknown inputs
            for input_obj in plan_run_inputs:
                if not any(plan_input.name == input_obj.name for plan_input in plan.plan_inputs):
                    logger().warning(f"Ignoring unknown plan input: {input_obj.name}")

            self.storage.save_plan_run(plan_run)

    def execute_plan_run_and_handle_clarifications(
        self,
        plan: Plan,
        plan_run: PlanRun,
    ) -> PlanRun:
        """Execute a plan run and handle any clarifications that are raised."""
        while plan_run.state not in [
            PlanRunState.COMPLETE,
            PlanRunState.FAILED,
        ]:
            plan_run = self._execute_plan_run(plan, plan_run)

            # If we don't have a clarification handler, return the plan run even if a clarification
            # has been raised
            if not self.execution_hooks.clarification_handler:
                return plan_run

            clarifications = plan_run.get_outstanding_clarifications()
            for clarification in clarifications:
                self.execution_hooks.clarification_handler.handle(
                    clarification=clarification,
                    on_resolution=lambda c, r: self.resolve_clarification(c, r) and None,
                    on_error=lambda c, r: self.error_clarification(c, r) and None,
                )

            if len(clarifications) > 0:
                # If clarifications are handled synchronously, we'll go through this immediately.
                # If they're handled asynchronously, we'll wait for the plan run to be ready.
                plan_run = self.wait_for_ready(plan_run)

        return plan_run

    def resolve_clarification(
        self,
        clarification: Clarification,
        response: object,
        plan_run: PlanRun | None = None,
    ) -> PlanRun:
        """Resolve a clarification updating the run state as needed.

        Args:
            clarification (Clarification): The clarification to resolve.
            response (object): The response to the clarification.
            plan_run (PlanRun | None): Optional - the plan run being updated.

        Returns:
            PlanRun: The updated PlanRun.

        """
        if plan_run is None:
            plan_run = self.storage.get_plan_run(clarification.plan_run_id)

        matched_clarification = next(
            (c for c in plan_run.outputs.clarifications if c.id == clarification.id),
            None,
        )

        if not matched_clarification:
            raise InvalidPlanRunStateError("Could not match clarification to run")

        matched_clarification.resolved = True
        matched_clarification.response = response

        if len(plan_run.get_outstanding_clarifications()) == 0:
            self._set_plan_run_state(plan_run, PlanRunState.READY_TO_RESUME)

        logger().info(
            f"Clarification resolved with response: {matched_clarification.response}",
        )

        logger().debug(
            f"Clarification resolved: {matched_clarification.model_dump_json(indent=4)}",
        )
        self.storage.save_plan_run(plan_run)
        return plan_run

    def error_clarification(
        self,
        clarification: Clarification,
        error: object,
        plan_run: PlanRun | None = None,
    ) -> PlanRun:
        """Mark that there was an error handling the clarification."""
        logger().error(
            f"Error handling clarification with guidance '{clarification.user_guidance}': {error}",
        )
        if plan_run is None:
            plan_run = self.storage.get_plan_run(clarification.plan_run_id)
        self._set_plan_run_state(plan_run, PlanRunState.FAILED)
        return plan_run

    def wait_for_ready(  # noqa: C901
        self,
        plan_run: PlanRun,
        max_retries: int = 6,
        backoff_start_time_seconds: int = 7 * 60,
        backoff_time_seconds: int = 2,
    ) -> PlanRun:
        """Wait for the run to be in a state that it can be re-plan_run.

        This is generally because there are outstanding clarifications that need to be resolved.

        Args:
            plan_run (PlanRun): The PlanRun to wait for.
            max_retries (int): The maximum number of retries to wait for the run to be ready
                after the backoff period starts.
            backoff_start_time_seconds (int): The time after which the backoff period starts.
            backoff_time_seconds (int): The time to wait between retries after the backoff period
                starts.

        Returns:
            PlanRun: The updated PlanRun once it is ready to be re-plan_run.

        Raises:
            InvalidRunStateError: If the run cannot be waited for.

        """
        start_time = time.time()
        tries = 0
        if plan_run.state not in [
            PlanRunState.IN_PROGRESS,
            PlanRunState.NOT_STARTED,
            PlanRunState.READY_TO_RESUME,
            PlanRunState.NEED_CLARIFICATION,
        ]:
            raise InvalidPlanRunStateError("Cannot wait for run that is not ready to run")

        # These states can continue straight away
        if plan_run.state in [
            PlanRunState.IN_PROGRESS,
            PlanRunState.NOT_STARTED,
            PlanRunState.READY_TO_RESUME,
        ]:
            return plan_run

        plan = self.storage.get_plan(plan_run.plan_id)
        while plan_run.state != PlanRunState.READY_TO_RESUME:
            plan_run = self.storage.get_plan_run(plan_run.id)
            current_step_clarifications = plan_run.get_clarifications_for_step()
            if tries >= max_retries:
                raise InvalidPlanRunStateError("Run is not ready to resume after max retries")

            # if we've waited longer than the backoff time, start the backoff period
            if time.time() - start_time > backoff_start_time_seconds:
                tries += 1
                backoff_time_seconds *= 2

            # wait a couple of seconds as we're long polling
            time.sleep(backoff_time_seconds)

            step = plan.steps[plan_run.current_step_index]
            next_tool = self._get_tool_for_step(step, plan_run)
            if next_tool:
                ready_response = next_tool.ready(
                    ToolRunContext(
                        end_user=self.initialize_end_user(plan_run.end_user_id),
                        plan_run_id=plan_run.id,
                        config=self.config,
                        clarifications=current_step_clarifications,
                    ),
                )
                logger().debug(f"Tool state for {next_tool.name} is ready={ready_response.ready}")
                if ready_response.ready:
                    for clarification in current_step_clarifications:
                        if clarification.category is ClarificationCategory.ACTION:
                            clarification.resolved = True
                            clarification.response = "complete"
                    if len(plan_run.get_outstanding_clarifications()) == 0:
                        self._set_plan_run_state(plan_run, PlanRunState.READY_TO_RESUME)
                else:
                    for clarification in current_step_clarifications:
                        logger().info(
                            f"Waiting for clarification {clarification.category} to be resolved",
                        )

            logger().info(f"New run state for {plan_run.id!s} is {plan_run.state!s}")

        logger().info(f"Run {plan_run.id!s} is ready to resume")

        return plan_run

    def _set_plan_run_state(self, plan_run: PlanRun, state: PlanRunState) -> None:
        """Set the state of a plan run and persist it to storage."""
        plan_run.state = state
        self.storage.save_plan_run(plan_run)

    def create_plan_run(
        self,
        plan: Plan,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | None = None,
    ) -> PlanRun:
        """Create a PlanRun from a Plan.

        Args:
            plan (Plan): The plan to create a plan run from.
            end_user (str | EndUser | None = None): The end user this plan run is for.
            plan_run_inputs (list[PlanInput] | None = None): The plan inputs for the
              plan run with their values.

        Returns:
            PlanRun: The created PlanRun object.

        """
        end_user = self.initialize_end_user(end_user)
        plan_run = PlanRun(
            plan_id=plan.id,
            state=PlanRunState.NOT_STARTED,
            end_user_id=end_user.external_id,
        )
        self._process_plan_input_values(plan, plan_run, plan_run_inputs)
        self.storage.save_plan_run(plan_run)
        return plan_run

    def _execute_plan_run(self, plan: Plan, plan_run: PlanRun) -> PlanRun:  # noqa: C901, PLR0912, PLR0915
        """Execute the run steps, updating the run state as needed.

        Args:
            plan (Plan): The plan to execute.
            plan_run (PlanRun): The plan run to execute.

        Returns:
            Run: The updated run after execution.

        """
        self._set_plan_run_state(plan_run, PlanRunState.IN_PROGRESS)

        dashboard_url = self.config.must_get("portia_dashboard_url", str)

        dashboard_message = (
            (
                f" View in your Portia AI dashboard: "
                f"{dashboard_url}/dashboard/plan-runs?plan_run_id={plan_run.id!s}"
            )
            if self.config.storage_class == StorageClass.CLOUD
            else ""
        )

        logger().info(
            f"Plan Run State is updated to {plan_run.state!s}.{dashboard_message}",
        )

        if self.execution_hooks.before_first_step_execution and plan_run.current_step_index == 0:
            self.execution_hooks.before_first_step_execution(
                ReadOnlyPlan.from_plan(plan), ReadOnlyPlanRun.from_plan_run(plan_run)
            )

        last_executed_step_output = self._get_last_executed_step_output(plan, plan_run)
        introspection_agent = self._get_introspection_agent()
        for index in range(plan_run.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            plan_run.current_step_index = index

            try:
                # Handle the introspection outcome
                (plan_run, pre_step_outcome) = self._handle_introspection_outcome(
                    introspection_agent=introspection_agent,
                    plan=plan,
                    plan_run=plan_run,
                    last_executed_step_output=last_executed_step_output,
                )
                if pre_step_outcome.outcome == PreStepIntrospectionOutcome.SKIP:
                    continue
                if pre_step_outcome.outcome != PreStepIntrospectionOutcome.CONTINUE:
                    self._log_final_output(plan_run, plan)
                    if (
                        self.execution_hooks.after_last_step_execution
                        and plan_run.outputs.final_output
                    ):
                        self.execution_hooks.after_last_step_execution(
                            ReadOnlyPlan.from_plan(plan),
                            ReadOnlyPlanRun.from_plan_run(plan_run),
                            plan_run.outputs.final_output,
                        )
                    return plan_run

                logger().info(
                    f"Executing step {index}: {step.task}",
                    plan=str(plan.id),
                    plan_run=str(plan_run.id),
                )

                if self.execution_hooks.before_step_execution:
                    self.execution_hooks.before_step_execution(
                        ReadOnlyPlan.from_plan(plan),
                        ReadOnlyPlanRun.from_plan_run(plan_run),
                        ReadOnlyStep.from_step(step),
                    )

                # we pass read only copies of the state to the agent so that the portia remains
                # responsible for handling the output of the agent and updating the state.
                agent = self._get_agent_for_step(
                    step=ReadOnlyStep.from_step(step),
                    plan_run=ReadOnlyPlanRun.from_plan_run(plan_run),
                )
                logger().debug(
                    f"Using agent: {type(agent).__name__}",
                    plan=str(plan.id),
                    plan_run=str(plan_run.id),
                )
                last_executed_step_output = agent.execute_sync()
            except Exception as e:  # noqa: BLE001 - We want to capture all failures here
                logger().error(f"Error executing step {index}: {e}")
                error_output = LocalDataValue(value=str(e))
                self._set_step_output(error_output, plan_run, step)
                plan_run.outputs.final_output = error_output
                self._set_plan_run_state(plan_run, PlanRunState.FAILED)
                logger().error(
                    "error: {error}",
                    error=e,
                    plan=str(plan.id),
                    plan_run=str(plan_run.id),
                )
                logger().debug(
                    f"Final run status: {plan_run.state!s}",
                    plan=str(plan.id),
                    plan_run=str(plan_run.id),
                )

                if self.execution_hooks.after_step_execution:
                    self.execution_hooks.after_step_execution(
                        ReadOnlyPlan.from_plan(plan),
                        ReadOnlyPlanRun.from_plan_run(plan_run),
                        ReadOnlyStep.from_step(step),
                        error_output,
                    )

                if self.execution_hooks.after_last_step_execution:
                    self.execution_hooks.after_last_step_execution(
                        ReadOnlyPlan.from_plan(plan),
                        ReadOnlyPlanRun.from_plan_run(plan_run),
                        plan_run.outputs.final_output,
                    )

                return plan_run
            else:
                self._set_step_output(last_executed_step_output, plan_run, step)
                logger().info(
                    f"Step output - {last_executed_step_output.get_summary()!s}",
                )

            if self.execution_hooks.after_step_execution:
                self.execution_hooks.after_step_execution(
                    ReadOnlyPlan.from_plan(plan),
                    ReadOnlyPlanRun.from_plan_run(plan_run),
                    ReadOnlyStep.from_step(step),
                    last_executed_step_output,
                )

            if self._raise_clarifications(plan_run, last_executed_step_output, plan):
                # No after_last_step_execution call here as the plan run will be resumed later
                return plan_run

            # persist at the end of each step
            self.storage.save_plan_run(plan_run)
            logger().debug(
                f"New PlanRun State: {plan_run.model_dump_json(indent=4)}",
            )

        if last_executed_step_output:
            plan_run.outputs.final_output = self._get_final_output(
                plan,
                plan_run,
                last_executed_step_output,
            )
        self._set_plan_run_state(plan_run, PlanRunState.COMPLETE)
        self._log_final_output(plan_run, plan)

        if self.execution_hooks.after_last_step_execution and plan_run.outputs.final_output:
            self.execution_hooks.after_last_step_execution(
                ReadOnlyPlan.from_plan(plan),
                ReadOnlyPlanRun.from_plan_run(plan_run),
                plan_run.outputs.final_output,
            )

        return plan_run

    def _log_final_output(self, plan_run: PlanRun, plan: Plan) -> None:
        logger().debug(
            f"Final run status: {plan_run.state!s}",
            plan=str(plan.id),
            plan_run=str(plan_run.id),
        )
        if plan_run.outputs.final_output:
            logger().info(
                f"Final output: {plan_run.outputs.final_output.get_summary()!s}",
            )

    def _get_last_executed_step_output(self, plan: Plan, plan_run: PlanRun) -> Output | None:
        """Get the output of the last executed step.

        Args:
            plan (Plan): The plan containing steps.
            plan_run (PlanRun): The plan run to get the output from.

        Returns:
            Output | None: The output of the last executed step.

        """
        return next(
            (
                plan_run.outputs.step_outputs[step.output]
                for i in range(plan_run.current_step_index, -1, -1)
                if i < len(plan.steps)
                and (step := plan.steps[i]).output in plan_run.outputs.step_outputs
                and (step_output := plan_run.outputs.step_outputs[step.output])
                and step_output.get_value() != PreStepIntrospectionOutcome.SKIP
            ),
            None,
        )

    def _handle_introspection_outcome(
        self,
        introspection_agent: BaseIntrospectionAgent,
        plan: Plan,
        plan_run: PlanRun,
        last_executed_step_output: Output | None,
    ) -> tuple[PlanRun, PreStepIntrospection]:
        """Handle the outcome of the pre-step introspection.

        Args:
            introspection_agent (BaseIntrospectionAgent): The introspection agent to use.
            plan (Plan): The plan being executed.
            plan_run (PlanRun): The plan run being executed.
            last_executed_step_output (Output | None): The output of the last step executed.

        Returns:
            tuple[PlanRun, PreStepIntrospectionOutcome]: The updated plan run and the
                outcome of the introspection.

        """
        current_step_index = plan_run.current_step_index
        step = plan.steps[current_step_index]
        if not step.condition:
            return (
                plan_run,
                PreStepIntrospection(
                    outcome=PreStepIntrospectionOutcome.CONTINUE,
                    reason="No condition to evaluate.",
                ),
            )

        logger().info(
            f"Evaluating condition for Step #{current_step_index}: #{step.condition}",
        )

        pre_step_outcome = introspection_agent.pre_step_introspection(
            plan=ReadOnlyPlan.from_plan(plan),
            plan_run=ReadOnlyPlanRun.from_plan_run(plan_run),
        )

        log_message = (
            f"Condition Evaluation Outcome for Step #{current_step_index} is "
            f"{pre_step_outcome.outcome.value}. "
            f"Reason: {pre_step_outcome.reason}",
        )

        logger().info(*log_message)

        match pre_step_outcome.outcome:
            case PreStepIntrospectionOutcome.SKIP:
                output = LocalDataValue(
                    value=SKIPPED_OUTPUT,
                    summary=pre_step_outcome.reason,
                )
                self._set_step_output(output, plan_run, step)
            case PreStepIntrospectionOutcome.COMPLETE:
                output = LocalDataValue(
                    value=COMPLETED_OUTPUT,
                    summary=pre_step_outcome.reason,
                )
                self._set_step_output(output, plan_run, step)
                if last_executed_step_output:
                    plan_run.outputs.final_output = self._get_final_output(
                        plan,
                        plan_run,
                        last_executed_step_output,
                    )
                self._set_plan_run_state(plan_run, PlanRunState.COMPLETE)
        return (plan_run, pre_step_outcome)

    def _get_planning_agent(self) -> BasePlanningAgent:
        """Get the planning_agent based on the configuration.

        Returns:
            BasePlanningAgent: The planning agent to be used for generating plans.

        """
        cls: type[BasePlanningAgent]
        match self.config.planning_agent_type:
            case PlanningAgentType.DEFAULT:
                cls = DefaultPlanningAgent

        return cls(self.config)

    def _get_final_output(self, plan: Plan, plan_run: PlanRun, step_output: Output) -> Output:
        """Get the final output and add summarization to it.

        Args:
            plan (Plan): The plan to execute.
            plan_run (PlanRun): The PlanRun to execute.
            step_output (Output): The output of the last step.

        """
        final_output = LocalDataValue(
            value=step_output.get_value(),
            summary=None,
        )

        try:
            summarizer = FinalOutputSummarizer(config=self.config)
            summary = summarizer.create_summary(
                plan_run=ReadOnlyPlanRun.from_plan_run(plan_run),
                plan=ReadOnlyPlan.from_plan(plan),
            )
            final_output.summary = summary

        except Exception as e:  # noqa: BLE001
            logger().warning(f"Error summarising run: {e}")

        return final_output

    def _raise_clarifications(self, plan_run: PlanRun, step_output: Output, plan: Plan) -> bool:
        """Update the plan run based on any clarifications raised.

        Args:
            plan_run (PlanRun): The PlanRun to execute.
            step_output (Output): The output of the last step.
            plan (Plan): The plan to execute.

        Returns:
            bool: True if clarification is needed and run execution should stop.

        """
        output_value = step_output.get_value()
        if isinstance(output_value, Clarification) or (
            isinstance(output_value, list)
            and len(output_value) > 0
            and all(isinstance(item, Clarification) for item in output_value)
        ):
            new_clarifications = (
                [output_value] if isinstance(output_value, Clarification) else output_value
            )
            for clarification in new_clarifications:
                clarification.step = plan_run.current_step_index
                logger().info(
                    "Clarification requested - category: %s, user_guidance: %r.",
                    clarification.category,
                    clarification.user_guidance,
                    extra={"plan": str(plan.id), "plan_run": str(plan_run.id)},
                )
                logger().debug(
                    f"Clarification requested: {clarification.model_dump_json(indent=4)}",
                )

            plan_run.outputs.clarifications = plan_run.outputs.clarifications + new_clarifications
            self._set_plan_run_state(plan_run, PlanRunState.NEED_CLARIFICATION)
            return True
        return False

    def _get_tool_for_step(self, step: Step, plan_run: PlanRun) -> Tool | None:
        if not step.tool_id:
            return None
        if step.tool_id == LLMTool.LLM_TOOL_ID:
            # Special case LLMTool so it doesn't need to be in all tool registries
            child_tool = LLMTool()
        else:
            child_tool = self.tool_registry.get_tool(step.tool_id)
        return ToolCallWrapper(
            child_tool=child_tool,
            storage=self.storage,
            plan_run=plan_run,
        )

    def _get_agent_for_step(
        self,
        step: Step,
        plan_run: PlanRun,
    ) -> BaseExecutionAgent:
        """Get the appropriate agent for executing a given step.

        Args:
            step (Step): The step for which the agent is needed.
            plan_run (PlanRun): The run associated with the step.

        Returns:
            BaseAgent: The agent to execute the step.

        """
        tool = self._get_tool_for_step(step, plan_run)
        cls: type[BaseExecutionAgent]
        match self.config.execution_agent_type:
            case ExecutionAgentType.ONE_SHOT:
                cls = OneShotAgent
            case ExecutionAgentType.DEFAULT:
                cls = DefaultExecutionAgent
        cls = OneShotAgent if isinstance(tool, LLMTool) else cls
        return cls(
            step,
            plan_run,
            self.config,
            self.storage,
            self.initialize_end_user(plan_run.end_user_id),
            tool,
            execution_hooks=self.execution_hooks,
        )

    def _log_replan_with_portia_cloud_tools(
        self,
        original_error: str,
        query: str,
        end_user: EndUser,
        example_plans: list[Plan] | None = None,
    ) -> None:
        """Generate a plan using Portia cloud tools for users who's plans fail without them."""
        unauthenticated_client = PortiaCloudClient.new_client(
            self.config,
            allow_unauthenticated=True,
        )
        portia_registry = PortiaToolRegistry(
            client=unauthenticated_client,
        ).with_default_tool_filter()
        cloud_registry = self.tool_registry + portia_registry
        tools = cloud_registry.match_tools(query)
        planning_agent = self._get_planning_agent()
        replan_outcome = planning_agent.generate_steps_or_error(
            query=query,
            tool_list=tools,
            end_user=end_user,
            examples=example_plans,
        )
        if not replan_outcome.error:
            tools_used = ", ".join([str(step.tool_id) for step in replan_outcome.steps])
            logger().error(
                f"Error in planning - {original_error.rstrip('.')}.\n"
                f"Replanning with Portia cloud tools would successfully generate a plan using "
                f"tools: {tools_used}.\n"
                f"Go to https://app.portialabs.ai to sign up.",
            )
            raise PlanError(
                "PORTIA_API_KEY is required to use Portia cloud tools.",
            ) from PlanError(original_error)

    def _get_introspection_agent(self) -> BaseIntrospectionAgent:
        return DefaultIntrospectionAgent(self.config, self.storage)

    def _set_step_output(self, output: Output, plan_run: PlanRun, step: Step) -> None:
        """Set the output for a step."""
        plan_run.outputs.step_outputs[step.output] = output
        self._persist_step_state(plan_run, step)

    def _persist_step_state(self, plan_run: PlanRun, step: Step) -> None:
        """Ensure the plan run state is persisted to storage."""
        step_output = plan_run.outputs.step_outputs[step.output]
        if (
            isinstance(step_output, LocalDataValue)
            and self.config.exceeds_output_threshold(
                step_output.serialize_value(),
            )
            # One-shot agent does not support pulling outputs from agent memory
            and self.config.execution_agent_type != ExecutionAgentType.ONE_SHOT
        ):
            step_output = self.storage.save_plan_run_output(step.output, step_output, plan_run.id)
            plan_run.outputs.step_outputs[step.output] = step_output

        self.storage.save_plan_run(plan_run)

    @staticmethod
    def _log_models(config: Config) -> None:
        """Log the models set in the configuration."""
        logger().debug("Portia Generative Models")
        for model in GenerativeModelsConfig.model_fields:
            getter = getattr(config, f"get_{model}")
            logger().debug(f"{model}: {getter()}")
