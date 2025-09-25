"""PortiaClient facade for eventual replacement of the monolithic Portia class.

This module provides the PortiaClient facade that will eventually replace
the monolithic Portia class. It uses the context pattern for dependency
management while maintaining the same public API.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from portia.config import Config
from portia.context import PortiaContext
from portia.end_user import EndUser
from portia.execution_hooks import ExecutionHooks
from portia.plan import Plan, PlanInput, PlanUUID
from portia.plan_run import PlanRun
from portia.telemetry.telemetry_service import BaseProductTelemetry
from portia.tool import Tool
from portia.tool_registry import ToolRegistry


class PortiaClient:
    """Facade for Portia functionality using context-based dependency management.

    This class provides the same public API as the original Portia class but uses
    the PortiaContext for dependency management. This is the first step toward
    delegation without changing behavior.
    """

    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
        execution_hooks: ExecutionHooks | None = None,
        telemetry: BaseProductTelemetry | None = None,
    ) -> None:
        """Initialize the PortiaClient with a PortiaContext.

        Args:
            config: The configuration to use. If not provided, the default configuration will be used.
            tools: The registry or list of tools to use. If not provided, the default tool registry will be used.
            execution_hooks: Hooks that can be used to modify or add extra functionality to plan runs.
            telemetry: Anonymous telemetry service.

        """
        self.context = PortiaContext(
            config=config,
            tools=tools,
            execution_hooks=execution_hooks,
            telemetry=telemetry,
        )

    def run(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> PlanRun:
        """End-to-end function to generate a plan and then execute it.

        This is the simplest way to plan and execute a query using the SDK.

        Args:
            query: The query to be executed.
            tools: List of tools to use for the query. If not provided all tools in the registry will be used.
            example_plans: Optional list of example plans or plan IDs.
            end_user: The end user for this plan run.
            plan_run_inputs: Provides input values for the run.
            structured_output_schema: The optional structured output schema for the query.
            use_cached_plan: Whether to use a cached plan if it exists.

        Returns:
            The run resulting from executing the query.

        """
        # TODO: Implement delegation to PortiaContext-based implementation
        # For now, this is a placeholder that will be implemented in the next phase
        raise NotImplementedError("PortiaClient.run will be implemented in the next phase")

    async def arun(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> PlanRun:
        """Async end-to-end function to generate a plan and then execute it.

        This is the simplest way to plan and execute a query using the SDK.

        Args:
            query: The query to be executed.
            tools: List of tools to use for the query. If not provided all tools in the registry will be used.
            example_plans: Optional list of example plans or plan IDs.
            end_user: The end user for this plan run.
            plan_run_inputs: Provides input values for the run.
            structured_output_schema: The optional structured output schema for the query.
            use_cached_plan: Whether to use a cached plan if it exists.

        Returns:
            The run resulting from executing the query.

        """
        # TODO: Implement delegation to PortiaContext-based implementation
        # For now, this is a placeholder that will be implemented in the next phase
        raise NotImplementedError("PortiaClient.arun will be implemented in the next phase")

    def plan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Generate a plan for the query.

        Args:
            query: The query to be planned.
            tools: List of tools to use for the query. If not provided all tools in the registry will be used.
            example_plans: Optional list of example plans or plan IDs.
            end_user: The end user for this plan.
            plan_run_inputs: Provides input values for the plan.
            structured_output_schema: The optional structured output schema for the query.
            use_cached_plan: Whether to use a cached plan if it exists.

        Returns:
            The generated plan.

        """
        # TODO: Implement delegation to PortiaContext-based implementation
        # For now, this is a placeholder that will be implemented in the next phase
        raise NotImplementedError("PortiaClient.plan will be implemented in the next phase")

    async def aplan(
        self,
        query: str,
        tools: list[Tool] | list[str] | None = None,
        example_plans: Sequence[Plan | PlanUUID | str] | None = None,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
        use_cached_plan: bool = False,
    ) -> Plan:
        """Async generate a plan for the query.

        Args:
            query: The query to be planned.
            tools: List of tools to use for the query. If not provided all tools in the registry will be used.
            example_plans: Optional list of example plans or plan IDs.
            end_user: The end user for this plan.
            plan_run_inputs: Provides input values for the plan.
            structured_output_schema: The optional structured output schema for the query.
            use_cached_plan: Whether to use a cached plan if it exists.

        Returns:
            The generated plan.

        """
        # TODO: Implement delegation to PortiaContext-based implementation
        # For now, this is a placeholder that will be implemented in the next phase
        raise NotImplementedError("PortiaClient.aplan will be implemented in the next phase")
