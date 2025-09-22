"""PortiaClient façade for lazy service composition.

This module provides a thin façade class that preserves the public API
while setting up lazy service instantiation infrastructure for future
service-based architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.portia import Portia as _LegacyPortia

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from portia.config import Config
    from portia.end_user import EndUser
    from portia.execution_hooks import ExecutionHooks
    from portia.plan import Plan, PlanInput, PlanUUID
    from portia.plan_run import PlanRun, PlanRunUUID
    from portia.telemetry.telemetry_service import BaseProductTelemetry
    from portia.tool import Tool
    from portia.tool_registry import ToolRegistry


class PortiaClient:
    """Portia client façade that preserves public API while enabling lazy service composition.

    This class serves as a thin wrapper around the existing Portia implementation,
    providing the same public interface while setting up infrastructure for lazy
    instantiation of services like PlanningService, PlanRunService, ExecutionEngine,
    ClarificationManager, and BuilderPlanExecutor.

    Initially delegates to existing Portia methods to maintain zero behavior change.
    Service wiring will be switched as services are implemented.
    """

    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
        execution_hooks: ExecutionHooks | None = None,
        telemetry: BaseProductTelemetry | None = None,
        # Future service overrides (currently unused but prepared for future use)
        planning_service: Any = None,  # noqa: ANN401
        plan_run_service: Any = None,  # noqa: ANN401
        execution_engine: Any = None,  # noqa: ANN401
        clarification_manager: Any = None,  # noqa: ANN401
        builder_plan_executor: Any = None,  # noqa: ANN401
    ) -> None:
        """Initialize the PortiaClient façade.

        Args:
            config: The configuration to initialize the Portia client.
            tools: The registry or list of tools to use.
            execution_hooks: Hooks for modifying or adding extra functionality.
            telemetry: Anonymous telemetry service.
            planning_service: Optional override for planning service (future use).
            plan_run_service: Optional override for plan run service (future use).
            execution_engine: Optional override for execution engine (future use).
            clarification_manager: Optional override for clarification manager (future use).
            builder_plan_executor: Optional override for builder plan executor (future use).

        """
        # Store service overrides for future lazy instantiation
        self._planning_service_override = planning_service
        self._plan_run_service_override = plan_run_service
        self._execution_engine_override = execution_engine
        self._clarification_manager_override = clarification_manager
        self._builder_plan_executor_override = builder_plan_executor

        # Currently delegate to existing Portia implementation
        self._legacy_portia = _LegacyPortia(
            config=config,
            tools=tools,
            execution_hooks=execution_hooks,
            telemetry=telemetry,
        )

    # Lazy service instantiation properties (prepared for future use)
    @property
    def _planning_service(self) -> Any:  # noqa: ANN401
        """Lazily instantiate PlanningService."""
        if not hasattr(self, "__planning_service"):
            if self._planning_service_override:
                self.__planning_service = self._planning_service_override
            else:
                # Future: instantiate PlanningService with PortiaContext
                # For now, return None as services aren't implemented yet
                self.__planning_service = None
        return self.__planning_service

    @property
    def _plan_run_service(self) -> Any:  # noqa: ANN401
        """Lazily instantiate PlanRunService."""
        if not hasattr(self, "__plan_run_service"):
            if self._plan_run_service_override:
                self.__plan_run_service = self._plan_run_service_override
            else:
                # Future: instantiate PlanRunService with PortiaContext
                self.__plan_run_service = None
        return self.__plan_run_service

    @property
    def _execution_engine(self) -> Any:  # noqa: ANN401
        """Lazily instantiate ExecutionEngine."""
        if not hasattr(self, "__execution_engine"):
            if self._execution_engine_override:
                self.__execution_engine = self._execution_engine_override
            else:
                # Future: instantiate ExecutionEngine with PortiaContext
                self.__execution_engine = None
        return self.__execution_engine

    @property
    def _clarification_manager(self) -> Any:  # noqa: ANN401
        """Lazily instantiate ClarificationManager."""
        if not hasattr(self, "__clarification_manager"):
            if self._clarification_manager_override:
                self.__clarification_manager = self._clarification_manager_override
            else:
                # Future: instantiate ClarificationManager with PortiaContext
                self.__clarification_manager = None
        return self.__clarification_manager

    @property
    def _builder_plan_executor(self) -> Any:  # noqa: ANN401
        """Lazily instantiate BuilderPlanExecutor."""
        if not hasattr(self, "__builder_plan_executor"):
            if self._builder_plan_executor_override:
                self.__builder_plan_executor = self._builder_plan_executor_override
            else:
                # Future: instantiate BuilderPlanExecutor with PortiaContext
                self.__builder_plan_executor = None
        return self.__builder_plan_executor

    # Public API methods - delegate to existing Portia implementation

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
        """End-to-end function to generate a plan and then execute it."""
        return self._legacy_portia.run(
            query=query,
            tools=tools,
            example_plans=example_plans,
            end_user=end_user,
            plan_run_inputs=plan_run_inputs,
            structured_output_schema=structured_output_schema,
            use_cached_plan=use_cached_plan,
        )

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
        """End-to-end function to generate a plan and then execute it (async)."""
        return await self._legacy_portia.arun(
            query=query,
            tools=tools,
            example_plans=example_plans,
            end_user=end_user,
            plan_run_inputs=plan_run_inputs,
            structured_output_schema=structured_output_schema,
            use_cached_plan=use_cached_plan,
        )

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
        """Generate a plan for executing a query."""
        return self._legacy_portia.plan(
            query=query,
            tools=tools,
            example_plans=example_plans,
            end_user=end_user,
            plan_run_inputs=plan_run_inputs,
            structured_output_schema=structured_output_schema,
            use_cached_plan=use_cached_plan,
        )

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
        """Generate a plan for executing a query (async)."""
        return await self._legacy_portia.aplan(
            query=query,
            tools=tools,
            example_plans=example_plans,
            end_user=end_user,
            plan_run_inputs=plan_run_inputs,
            structured_output_schema=structured_output_schema,
            use_cached_plan=use_cached_plan,
        )

    def run_plan(
        self,
        plan: Plan | PlanUUID | str,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
    ) -> PlanRun:
        """Execute an existing plan."""
        return self._legacy_portia.run_plan(
            plan=plan,
            end_user=end_user,
            plan_run_inputs=plan_run_inputs,
            structured_output_schema=structured_output_schema,
        )

    async def arun_plan(
        self,
        plan: Plan | PlanUUID | str,
        end_user: str | EndUser | None = None,
        plan_run_inputs: list[PlanInput] | list[dict[str, str]] | dict[str, str] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
    ) -> PlanRun:
        """Execute an existing plan (async)."""
        return await self._legacy_portia.arun_plan(
            plan=plan,
            end_user=end_user,
            plan_run_inputs=plan_run_inputs,
            structured_output_schema=structured_output_schema,
        )

    def resume(
        self,
        plan_run: PlanRun | PlanRunUUID | str,
        end_user: str | EndUser | None = None,
    ) -> PlanRun:
        """Resume execution of a paused plan run."""
        return self._legacy_portia.resume(
            plan_run=plan_run,
            end_user=end_user,
        )

    async def aresume(
        self,
        plan_run: PlanRun | PlanRunUUID | str,
        end_user: str | EndUser | None = None,
    ) -> PlanRun:
        """Resume execution of a paused plan run (async)."""
        return await self._legacy_portia.aresume(
            plan_run=plan_run,
            end_user=end_user,
        )

    # Builder helper methods (delegate to existing implementation)
    async def run_builder_plan(self, plan: Any, **kwargs: Any) -> PlanRun:  # noqa: ANN401
        """Run a builder plan."""
        return await self._legacy_portia.run_builder_plan(plan, **kwargs)

    async def resume_builder_plan(self, plan_run: PlanRun, **kwargs: Any) -> PlanRun:
        """Resume a builder plan."""
        return await self._legacy_portia.resume_builder_plan(plan_run, **kwargs)

    # Delegate any other attributes to the legacy implementation
    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate any missing attributes to the legacy Portia implementation."""
        return getattr(self._legacy_portia, name)
