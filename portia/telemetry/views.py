"""Portia telemetry views."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BaseTelemetryEvent(ABC):
    """Base class for all telemetry events.

    This abstract class defines the interface that all telemetry events must implement.
    It provides a common structure for event name and properties.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the telemetry event.

        Returns:
            str: The name of the telemetry event.

        """

    @property
    def properties(self) -> dict[str, Any]:
        """Get the properties of the telemetry event.

        Returns:
            dict[str, Any]: A dictionary containing all properties of the event,
                           excluding the 'name' property.

        """
        return {k: v for k, v in asdict(self).items() if k != "name"}  # pragma: no cover


@dataclass
class PortiaFunctionCallTelemetryEvent(BaseTelemetryEvent):
    """Telemetry event for tracking Portia function calls.

    Attributes:
        function_name: The name of the function being called.
        function_call_details: Additional details about the function call.

    """

    function_name: str
    function_call_details: dict[str, Any]
    name: str = "portia_function_call"  # type: ignore reportIncompatibleMethodOverride


@dataclass
class ToolCallTelemetryEvent(BaseTelemetryEvent):
    """Telemetry event for tracking tool calls.

    Attributes:
        tool_id: The identifier of the tool being called, if any.

    """

    tool_id: str | None
    name: str = "tool_call"  # type: ignore reportIncompatibleMethodOverride


@dataclass
class PlanV2StepExecutionTelemetryEvent(BaseTelemetryEvent):
    """Telemetry event for tracking PlanV2 step execution.

    Attributes:
        step_type: The type of the step being executed.
        success: Whether the step execution was successful.
        tool_id: The identifier of the tool being used, if any.

    """

    step_type: str
    success: bool
    tool_id: str | None
    name: str = "plan_v2_step_execution"  # type: ignore reportIncompatibleMethodOverride


@dataclass
class PlanV2BuildTelemetryEvent(BaseTelemetryEvent):
    """Telemetry event for tracking PlanV2 builds.

    Attributes:
        plan_length: The number of steps in the plan.
        step_type_counts: A dictionary mapping step types to their counts in the plan.

    """

    plan_length: int
    step_type_counts: dict[str, int]
    name: str = "plan_v2_build"  # type: ignore reportIncompatibleMethodOverride


@dataclass
class LLMToolUsageTelemetryEvent(BaseTelemetryEvent):
    """Telemetry event for tracking LLM tool usage.

    Attributes:
        model: The model being used.
        sync: Whether the tool was called synchronously.

    """

    model: str | None
    sync: bool
    name: str = "llm_tool_usage"  # type: ignore reportIncompatibleMethodOverride


@dataclass
class ExecutionAgentUsageTelemetryEvent(BaseTelemetryEvent):
    """Telemetry event for tracking execution agent usage.

    Attributes:
        agent_type: The type of the execution agent (e.g., "one_shot", "default").
        model: The model being used.
        sync: Whether the agent was called synchronously.
        tool_id: The identifier of the tool being used, if any.

    """

    agent_type: str
    model: str | None
    sync: bool
    tool_id: str | None
    name: str = "execution_agent_usage"  # type: ignore reportIncompatibleMethodOverride
