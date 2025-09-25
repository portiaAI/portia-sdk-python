"""Planning service for orchestrating plan generation in Portia.

This module contains the PlanningService class that handles all planning-related
operations, including plan generation, example plan resolution, and input coercion.
It provides both synchronous and asynchronous interfaces for plan generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.cloud import PortiaCloudClient
from portia.config import Config, PlanningAgentType
from portia.end_user import EndUser
from portia.errors import PlanError, PlanNotFoundError, StorageError
from portia.logger import logger
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from portia.telemetry.views import PortiaFunctionCallTelemetryEvent
from portia.tool import Tool
from portia.tool_registry import DefaultToolRegistry, PortiaToolRegistry, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from portia.planning_agents.base_planning_agent import BasePlanningAgent
    from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
    from portia.telemetry.telemetry_service import BaseProductTelemetry


class PlanningService:
    """Service class for handling planning orchestration in Portia.

    This class encapsulates all planning-related functionality including:
    - Plan generation (sync and async)
    - Example plan resolution
    - Plan input coercion
    - Telemetry and logging for planning operations
    """

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry,
        storage: InMemoryStorage | DiskFileStorage | PortiaCloudStorage,
        telemetry: BaseProductTelemetry,
    ) -> None:
        """Initialize the PlanningService.

        Args:
            config: The Portia configuration
            tool_registry: Registry for managing available tools
            storage: Storage backend for plans
            telemetry: Telemetry service for tracking operations

        """
        self.config = config
        self.tool_registry = tool_registry
        self.storage = storage
        self.telemetry = telemetry

    def plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Generate a plan synchronously.

        Args:
            query: The query to generate the plan for
            tools: List of tools to use for the query
            example_plans: Optional list of example plans or plan IDs
            end_user: The optional end user for this plan
            plan_inputs: Optional list of inputs required for the plan
            structured_output_schema: Optional structured output schema for the query
            use_cached_plan: Whether to use a cached plan if it exists

        Returns:
            Plan: The generated plan

        Raises:
            PlanError: If there is an error while generating the plan

        """
        self.telemetry.capture(
            PortiaFunctionCallTelemetryEvent(
                function_name="portia_plan",
                function_call_details={
                    "tools": (
                        ",".join([tool.id if isinstance(tool, Tool) else tool for tool in tools])
                        if tools
                        else None
                    ),
                    "example_plans_provided": example_plans is not None,
                    "end_user_provided": end_user is not None,
                    "plan_inputs_provided": plan_inputs is not None,
                },
            )
        )
        return self._plan(
            query,
            tools,
            example_plans,
            end_user,
            plan_inputs,
            structured_output_schema,
            use_cached_plan,
        )

    async def aplan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Generate a plan asynchronously.

        Args:
            query: The query to generate the plan for
            tools: List of tools to use for the query
            example_plans: Optional list of example plans or plan IDs
            end_user: The optional end user for this plan
            plan_inputs: Optional list of inputs required for the plan
            structured_output_schema: Optional structured output schema for the query
            use_cached_plan: Whether to use a cached plan if it exists

        Returns:
            Plan: The generated plan

        Raises:
            PlanError: If there is an error while generating the plan

        """
        self.telemetry.capture(
            PortiaFunctionCallTelemetryEvent(
                function_name="portia_aplan",
                function_call_details={
                    "tools": (
                        ",".join([tool.id if isinstance(tool, Tool) else tool for tool in tools])
                        if tools
                        else None
                    ),
                    "example_plans_provided": example_plans is not None,
                    "end_user_provided": end_user is not None,
                    "plan_inputs_provided": plan_inputs is not None,
                },
            )
        )
        return await self._aplan(
            query,
            tools,
            example_plans,
            end_user,
            plan_inputs,
            structured_output_schema,
            use_cached_plan,
        )

    async def generate_plan_async(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Primary async method for generating plans.

        This is the main async interface that both sync and async methods use.

        Args:
            query: The query to generate the plan for
            tools: List of tools to use for the query
            example_plans: Optional list of example plans or plan IDs
            end_user: The optional end user for this plan
            plan_inputs: Optional list of inputs required for the plan
            structured_output_schema: Optional structured output schema for the query
            use_cached_plan: Whether to use a cached plan if it exists

        Returns:
            Plan: The generated plan

        Raises:
            PlanError: If there is an error while generating the plan

        """
        return await self._aplan(
            query,
            tools,
            example_plans,
            end_user,
            plan_inputs,
            structured_output_schema,
            use_cached_plan,
        )

    def _plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Implement synchronous planning logic.

        This is used when we're already in an event loop and can't use asyncio.run().

        Args:
            query: The query to generate the plan for
            tools: List of tools to use for the query
            example_plans: Optional list of example plans or plan IDs
            end_user: The optional end user for this plan
            plan_inputs: Optional list of inputs required for the plan
            structured_output_schema: Optional structured output schema for the query
            use_cached_plan: Whether to use a cached plan if it exists

        Returns:
            Plan: The plan for executing the query

        Raises:
            PlanError: If there is an error while generating the plan

        """
        if use_cached_plan:
            try:
                return self.storage.get_plan_by_query(query)
            except StorageError as e:
                logger().warning(f"Error getting cached plan. Using new plan instead: {e}")

        if isinstance(tools, list):
            tools = [
                self.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.tool_registry.match_tools(query)

        resolved_example_plans = self._resolve_example_plans(example_plans)

        end_user = self._initialize_end_user(end_user)
        logger().info(f"Running planning_agent for query - {query}")
        planning_agent = self._get_planning_agent()
        coerced_plan_inputs = self._coerce_plan_inputs(plan_inputs)

        outcome = planning_agent.generate_steps_or_error(
            query=query,
            tool_list=tools,
            end_user=end_user,
            examples=resolved_example_plans,
            plan_inputs=coerced_plan_inputs,
        )

        if outcome.error:
            self._log_replan_with_portia_cloud_tools(
                outcome.error,
                query,
                end_user,
                resolved_example_plans,
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
            structured_output_schema=structured_output_schema,
        )

        self.storage.save_plan(plan)
        logger().info(
            f"Plan created with {len(plan.steps)} steps",
            plan=str(plan.id),
        )
        logger().debug(plan.pretty_print())

        return plan

    async def _aplan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_inputs: list[PlanInput] | list[dict[str, str]] | list[str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Async implementation of planning logic.

        This is the core async implementation that both sync and async methods use.

        Args:
            query: The query to generate the plan for
            tools: List of tools to use for the query
            example_plans: Optional list of example plans or plan IDs
            end_user: The optional end user for this plan
            plan_inputs: Optional list of inputs required for the plan
            structured_output_schema: Optional structured output schema for the query
            use_cached_plan: Whether to use a cached plan if it exists

        Returns:
            Plan: The plan for executing the query

        Raises:
            PlanError: If there is an error while generating the plan

        """
        if use_cached_plan:
            try:
                return await self.storage.aget_plan_by_query(query)
            except StorageError as e:
                logger().warning(f"Error getting cached plan. Using new plan instead: {e}")

        if isinstance(tools, list):
            tools = [
                self.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.tool_registry.match_tools(query)

        resolved_example_plans = await self._aresolve_example_plans(example_plans)

        end_user = await self._ainitialize_end_user(end_user)
        logger().info(f"Running planning_agent for query - {query}")
        planning_agent = self._get_planning_agent()
        coerced_plan_inputs = self._coerce_plan_inputs(plan_inputs)
        outcome = await planning_agent.agenerate_steps_or_error(
            query=query,
            tool_list=tools,
            end_user=end_user,
            examples=resolved_example_plans,
            plan_inputs=coerced_plan_inputs,
        )

        if outcome.error:
            self._log_replan_with_portia_cloud_tools(
                outcome.error,
                query,
                end_user,
                resolved_example_plans,
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
            structured_output_schema=structured_output_schema,
        )

        await self.storage.asave_plan(plan)
        logger().info(
            f"Plan created with {len(plan.steps)} steps",
            plan=str(plan.id),
        )
        logger().debug(plan.pretty_print())

        return plan

    def _resolve_example_plans(
        self, example_plans: Sequence[Plan | PlanUUID | str] | None
    ) -> list[Plan] | None:
        """Resolve example plans from Plan objects, PlanUUIDs and planID strings.

        Args:
            example_plans: List of example plans or plan IDs
                - Plan objects are used directly
                - PlanUUID objects are loaded from storage
                - String objects must be plan ID strings (starting with "plan-")

        Returns:
            List of resolved Plan objects, or None if input was None

        Raises:
            PlanNotFoundError: If a plan ID cannot be found in storage
            ValueError: If a string is not a plan ID string
            TypeError: If an invalid type is provided

        """
        if example_plans is None:
            return None

        resolved_plans = []
        for example_plan in example_plans:
            resolved_plan = self._resolve_single_example_plan(example_plan)
            resolved_plans.append(resolved_plan)

        return resolved_plans

    async def _aresolve_example_plans(
        self, example_plans: Sequence[Plan | PlanUUID | str] | None
    ) -> list[Plan] | None:
        """Resolve example plans from Plan objects, PlanUUIDs and planID strings.

        Args:
            example_plans: List of example plans or plan IDs
                - Plan objects are used directly
                - PlanUUID objects are loaded from storage
                - String objects must be plan ID strings (starting with "plan-")

        Returns:
            List of resolved Plan objects, or None if input was None

        Raises:
            PlanNotFoundError: If a plan ID cannot be found in storage
            ValueError: If a string is not a plan ID string
            TypeError: If an invalid type is provided

        """
        if example_plans is None:
            return None

        resolved_plans = []
        for example_plan in example_plans:
            resolved_plan = await self._aresolve_single_example_plan(example_plan)
            resolved_plans.append(resolved_plan)

        return resolved_plans

    def _resolve_single_example_plan(self, example_plan: Plan | PlanUUID | str) -> Plan:
        """Resolve a single example plan from various input types."""
        if isinstance(example_plan, Plan):
            return example_plan
        if isinstance(example_plan, PlanUUID):
            return self._load_plan_by_uuid(example_plan)
        if isinstance(example_plan, str):
            return self._resolve_string_example_plan(example_plan)
        raise TypeError(
            f"Invalid example plan type: {type(example_plan)}. Expected Plan, PlanUUID, or str."
        )

    async def _aresolve_single_example_plan(self, example_plan: Plan | PlanUUID | str) -> Plan:
        """Resolve a single example plan from various input types asynchronously."""
        if isinstance(example_plan, Plan):
            return example_plan
        if isinstance(example_plan, PlanUUID):
            return await self._aload_plan_by_uuid(example_plan)
        if isinstance(example_plan, str):
            return await self._aresolve_string_example_plan(example_plan)
        raise TypeError(
            f"Invalid example plan type: {type(example_plan)}. Expected Plan, PlanUUID, or str."
        )

    def _load_plan_by_uuid(self, plan_uuid: PlanUUID) -> Plan:
        """Load a plan from storage by UUID."""
        try:
            return self.storage.get_plan(plan_uuid)
        except Exception as e:
            raise PlanNotFoundError(plan_uuid) from e

    async def _aload_plan_by_uuid(self, plan_uuid: PlanUUID) -> Plan:
        """Load a plan from storage by UUID asynchronously."""
        try:
            return await self.storage.aget_plan(plan_uuid)
        except Exception as e:
            raise PlanNotFoundError(plan_uuid) from e

    def _resolve_string_example_plan(self, example_plan: str) -> Plan:
        """Resolve a string example plan - must be a plan ID string."""
        # Only support plan ID strings, not query strings
        if not example_plan.startswith("plan-"):
            raise ValueError(
                f"String '{example_plan}' must be a plan ID (starting with 'plan-'). "
                "Query strings are not supported."
            )

        plan_uuid = PlanUUID.from_string(example_plan)
        try:
            return self._load_plan_by_uuid(plan_uuid)
        except Exception as e:
            raise PlanNotFoundError(plan_uuid) from e

    async def _aresolve_string_example_plan(self, example_plan: str) -> Plan:
        """Resolve a string example plan - must be a plan ID string."""
        # Only support plan ID strings, not query strings
        if not example_plan.startswith("plan-"):
            raise ValueError(
                f"String '{example_plan}' must be a plan ID (starting with 'plan-'). "
                "Query strings are not supported."
            )

        plan_uuid = PlanUUID.from_string(example_plan)
        try:
            return await self._aload_plan_by_uuid(plan_uuid)
        except Exception as e:
            raise PlanNotFoundError(plan_uuid) from e

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

    def _initialize_end_user(self, end_user: str | EndUser | None = None) -> EndUser:
        """Handle initializing the end_user based on the provided type."""
        default_external_id = "portia:default_user"
        if isinstance(end_user, str):
            if end_user == "":
                end_user = default_external_id
            return EndUser(
                external_id=end_user,
                name=None,
            )
        if isinstance(end_user, EndUser):
            return end_user
        if end_user is None:
            return EndUser(
                external_id=default_external_id,
                name=None,
            )
        raise ValueError("end_user must be a string, EndUser, or None")

    async def _ainitialize_end_user(self, end_user: str | EndUser | None = None) -> EndUser:
        """Handle initializing the end_user based on the provided type."""
        default_external_id = "portia:default_user"
        if isinstance(end_user, str):
            if end_user == "":
                end_user = default_external_id
            return EndUser(
                external_id=end_user,
                name=None,
            )
        if isinstance(end_user, EndUser):
            return end_user
        if end_user is None:
            return EndUser(
                external_id=default_external_id,
                name=None,
            )
        raise ValueError("end_user must be a string, EndUser, or None")

    def _log_replan_with_portia_cloud_tools(
        self,
        original_error: str,
        query: str,
        end_user: EndUser,
        example_plans: list[Plan] | None = None,
    ) -> None:
        """Generate a plan using Portia cloud tools for users who's plans fail without them."""
        if not isinstance(self.tool_registry, DefaultToolRegistry) or self.config.portia_api_key:
            return
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

    def _get_planning_agent(self) -> BasePlanningAgent:
        """Get the planning_agent based on the configuration.

        Returns:
            BasePlanningAgent: The planning agent to be used for generating plans

        """
        cls: type[BasePlanningAgent]
        match self.config.planning_agent_type:
            case PlanningAgentType.DEFAULT:
                cls = DefaultPlanningAgent
            case _:
                raise ValueError(f"Unknown planning agent type: {self.config.planning_agent_type}")
        return cls(self.config, self.storage)
