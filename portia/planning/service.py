"""Planning service for generating and managing plans.

This module contains the PlanningService class, which is responsible for orchestrating
the planning agents and generating plans from user queries. It handles:
- Resolving tool lists for queries
- Loading and resolving example plans
- Orchestrating planning agents to generate plan steps
- Persisting plans to storage
- Handling end user initialization

The PlanningService is extracted from the monolithic Portia class to provide
a focused, single-responsibility service for plan generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.cloud import PortiaCloudClient
from portia.config import PlanningAgentType
from portia.end_user import EndUser
from portia.errors import PlanError, PlanNotFoundError, StorageError
from portia.logger import logger
from portia.plan import Plan, PlanContext, PlanInput, PlanUUID
from portia.planning_agents.default_planning_agent import DefaultPlanningAgent
from portia.tool_registry import DefaultToolRegistry, PortiaToolRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from portia.context import PortiaContext
    from portia.planning_agents.base_planning_agent import BasePlanningAgent
    from portia.tool import Tool


class PlanningService:
    """Service for generating and managing plans.

    The PlanningService is the primary orchestrator for planning agents. It takes a user
    query and generates an executable plan by:
    1. Resolving the list of available tools (either from explicit input or by matching)
    2. Loading example plans for few-shot learning
    3. Initializing the end user context
    4. Invoking the planning agent to generate steps
    5. Persisting the resulting plan to storage

    The service relies on a PortiaContext to access shared dependencies like storage,
    tool registry, configuration, and telemetry.

    Attributes:
        context: The shared PortiaContext providing access to dependencies.

    """

    def __init__(self, context: PortiaContext) -> None:
        """Initialize the PlanningService.

        Args:
            context: The PortiaContext providing access to storage, tool registry,
                config, and telemetry.

        """
        self.context = context

    def generate_plan(
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

        This method orchestrates the planning process by resolving tools, loading example
        plans, and invoking the planning agent to generate a plan for the given query.

        Args:
            query: The query to generate the plan for.
            tools: List of tools to use for the query. If not provided, all tools
                in the registry will be matched against the query.
            example_plans: Optional list of example plans or plan IDs for few-shot
                learning. This can include Plan objects, PlanUUID objects, or plan ID
                strings (starting with "plan-"). Plan IDs will be loaded from storage.
                If not provided, a default set of example plans will be used.
            end_user: The optional end user for this plan.
            plan_inputs: Optional list of inputs required for the plan. This can be
                a list of PlanInput objects, a list of dicts with keys "name" and
                "description" (optional), or a list of plan run input names. If a value
                is provided with a PlanInput object or in a dictionary, it will be
                ignored as values are only used when running the plan.
            structured_output_schema: The optional structured output schema for the
                query. This is passed on to plan runs created from this plan but will
                not be stored with the plan itself if using cloud storage and must be
                re-attached to the plan run if using cloud storage.
            use_cached_plan: Whether to use a cached plan if it exists.

        Returns:
            The generated Plan object.

        Raises:
            PlanError: If there is an error while generating the plan.

        """
        if use_cached_plan:
            try:
                return self.context.storage.get_plan_by_query(query)
            except StorageError as e:
                logger().warning(f"Error getting cached plan. Using new plan instead: {e}")

        # Resolve tools
        if isinstance(tools, list):
            tools = [
                self.context.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.context.tool_registry.match_tools(query)

        # Resolve example plans
        resolved_example_plans = self._resolve_example_plans(example_plans)

        # Initialize end user
        end_user_obj = self._initialize_end_user(end_user)

        logger().info(f"Running planning_agent for query - {query}")

        # Get planning agent
        planning_agent = self._get_planning_agent()

        # Coerce plan inputs
        coerced_plan_inputs = self._coerce_plan_inputs(plan_inputs)

        # Generate steps
        outcome = planning_agent.generate_steps_or_error(
            query=query,
            tool_list=tools,
            end_user=end_user_obj,
            examples=resolved_example_plans,
            plan_inputs=coerced_plan_inputs,
        )

        if outcome.error:
            self._log_replan_with_portia_cloud_tools(
                outcome.error,
                query,
                end_user_obj,
                resolved_example_plans,
            )
            logger().error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)

        # Create plan
        plan = Plan(
            plan_context=PlanContext(
                query=query,
                tool_ids=[tool.id for tool in tools],
            ),
            steps=outcome.steps,
            plan_inputs=coerced_plan_inputs or [],
            structured_output_schema=structured_output_schema,
        )

        # Save plan
        self.context.storage.save_plan(plan)
        logger().info(
            f"Plan created with {len(plan.steps)} steps",
            plan=str(plan.id),
        )
        logger().debug(plan.pretty_print())

        return plan

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
        """Generate a plan asynchronously.

        This is the async version of generate_plan. It performs the same logic but uses
        async methods for storage and end user operations.

        Args:
            query: The query to generate the plan for.
            tools: List of tools to use for the query. If not provided, all tools
                in the registry will be matched against the query.
            example_plans: Optional list of example plans or plan IDs for few-shot
                learning. This can include Plan objects, PlanUUID objects, or plan ID
                strings (starting with "plan-"). Plan IDs will be loaded from storage.
                If not provided, a default set of example plans will be used.
            end_user: The optional end user for this plan.
            plan_inputs: Optional list of inputs required for the plan. This can be
                a list of PlanInput objects, a list of dicts with keys "name" and
                "description" (optional), or a list of plan run input names. If a value
                is provided with a PlanInput object or in a dictionary, it will be
                ignored as values are only used when running the plan.
            structured_output_schema: The optional structured output schema for the
                query. This is passed on to plan runs created from this plan but will
                not be stored with the plan itself if using cloud storage and must be
                re-attached to the plan run if using cloud storage.
            use_cached_plan: Whether to use a cached plan if it exists.

        Returns:
            The generated Plan object.

        Raises:
            PlanError: If there is an error while generating the plan.

        """
        if use_cached_plan:
            try:
                return await self.context.storage.aget_plan_by_query(query)
            except StorageError as e:
                logger().warning(f"Error getting cached plan. Using new plan instead: {e}")

        # Resolve tools
        if isinstance(tools, list):
            tools = [
                self.context.tool_registry.get_tool(tool) if isinstance(tool, str) else tool
                for tool in tools
            ]

        if not tools:
            tools = self.context.tool_registry.match_tools(query)

        # Resolve example plans
        resolved_example_plans = await self._aresolve_example_plans(example_plans)

        # Initialize end user
        end_user_obj = await self._ainitialize_end_user(end_user)

        logger().info(f"Running planning_agent for query - {query}")

        # Get planning agent
        planning_agent = self._get_planning_agent()

        # Coerce plan inputs
        coerced_plan_inputs = self._coerce_plan_inputs(plan_inputs)

        # Generate steps
        outcome = await planning_agent.agenerate_steps_or_error(
            query=query,
            tool_list=tools,
            end_user=end_user_obj,
            examples=resolved_example_plans,
            plan_inputs=coerced_plan_inputs,
        )

        if outcome.error:
            self._log_replan_with_portia_cloud_tools(
                outcome.error,
                query,
                end_user_obj,
                resolved_example_plans,
            )
            logger().error(f"Error in planning - {outcome.error}")
            raise PlanError(outcome.error)

        # Create plan
        plan = Plan(
            plan_context=PlanContext(
                query=query,
                tool_ids=[tool.id for tool in tools],
            ),
            steps=outcome.steps,
            plan_inputs=coerced_plan_inputs or [],
            structured_output_schema=structured_output_schema,
        )

        # Save plan
        await self.context.storage.asave_plan(plan)
        logger().info(
            f"Plan created with {len(plan.steps)} steps",
            plan=str(plan.id),
        )
        logger().debug(plan.pretty_print())

        return plan

    def _initialize_end_user(self, end_user: str | EndUser | None = None) -> EndUser:
        """Handle initializing the end_user based on the provided type."""
        default_external_id = "portia:default_user"
        if isinstance(end_user, str):
            if end_user == "":
                end_user = default_external_id
            end_user_instance = self.context.storage.get_end_user(external_id=end_user)
            if end_user_instance:
                return end_user_instance
            end_user_instance = EndUser(external_id=end_user or default_external_id)
            return self.context.storage.save_end_user(end_user_instance)

        if not end_user:
            end_user = EndUser(external_id=default_external_id)
            return self.context.storage.save_end_user(end_user)

        return self.context.storage.save_end_user(end_user)

    async def _ainitialize_end_user(self, end_user: str | EndUser | None = None) -> EndUser:
        """Handle initializing the end_user based on the provided type (async)."""
        default_external_id = "portia:default_user"
        if isinstance(end_user, str):
            if end_user == "":
                end_user = default_external_id
            end_user_instance = await self.context.storage.aget_end_user(external_id=end_user)
            if end_user_instance:
                return end_user_instance
            end_user_instance = EndUser(external_id=end_user or default_external_id)
            return await self.context.storage.asave_end_user(end_user_instance)

        if not end_user:
            end_user = EndUser(external_id=default_external_id)
            return await self.context.storage.asave_end_user(end_user)

        return await self.context.storage.asave_end_user(end_user)

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

    def _resolve_example_plans(
        self, example_plans: Sequence[Plan | PlanUUID | str] | None
    ) -> list[Plan] | None:
        """Resolve example plans from Plan objects, PlanUUIDs and planID strings.

        Args:
            example_plans: List of example plans or plan IDs.
                - Plan objects are used directly
                - PlanUUID objects are loaded from storage
                - String objects must be plan ID strings (starting with "plan-")

        Returns:
            List of resolved Plan objects, or None if input was None.

        Raises:
            PlanNotFoundError: If a plan ID cannot be found in storage.
            ValueError: If a string is not a plan ID string.
            TypeError: If an invalid type is provided.

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
        """Resolve example plans from Plan objects, PlanUUIDs and planID strings (async).

        Args:
            example_plans: List of example plans or plan IDs.
                - Plan objects are used directly
                - PlanUUID objects are loaded from storage
                - String objects must be plan ID strings (starting with "plan-")

        Returns:
            List of resolved Plan objects, or None if input was None.

        Raises:
            PlanNotFoundError: If a plan ID cannot be found in storage.
            ValueError: If a string is not a plan ID string.
            TypeError: If an invalid type is provided.

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
        """Resolve a single example plan from various input types (async)."""
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
            return self.context.storage.get_plan(plan_uuid)
        except Exception as e:
            raise PlanNotFoundError(plan_uuid) from e

    async def _aload_plan_by_uuid(self, plan_uuid: PlanUUID) -> Plan:
        """Load a plan from storage by UUID asynchronously."""
        try:
            return await self.context.storage.aget_plan(plan_uuid)
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
        """Resolve a string example plan - must be a plan ID string (async)."""
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

    def _get_planning_agent(self) -> BasePlanningAgent:
        """Get the planning_agent based on the configuration.

        Returns:
            The planning agent to be used for generating plans.

        """
        cls: type[BasePlanningAgent]
        match self.context.config.planning_agent_type:
            case PlanningAgentType.DEFAULT:
                cls = DefaultPlanningAgent

        return cls(self.context.config)

    def _log_replan_with_portia_cloud_tools(
        self,
        original_error: str,
        query: str,
        end_user: EndUser,
        example_plans: list[Plan] | None = None,
    ) -> None:
        """Generate a plan using Portia cloud tools for users who's plans fail without them."""
        if (
            not isinstance(self.context.tool_registry, DefaultToolRegistry)
            or self.context.config.portia_api_key
        ):
            return
        unauthenticated_client = PortiaCloudClient.new_client(
            self.context.config,
            allow_unauthenticated=True,
        )
        portia_registry = PortiaToolRegistry(
            client=unauthenticated_client,
        ).with_default_tool_filter()
        cloud_registry = self.context.tool_registry + portia_registry
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
                f"However, this task looks possible with Portia Cloud Tools: {tools_used}.\n"
                "See https://docs.portia.sh/latest/concepts/tools for more information."
            )
