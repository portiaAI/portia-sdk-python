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
        return {k: v for k, v in asdict(self).items() if k != "name"}


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
