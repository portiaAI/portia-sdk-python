"""No-op telemetry service (telemetry removed)."""

from abc import ABC, abstractmethod

from portia.common import singleton
from portia.telemetry.views import BaseTelemetryEvent


class BaseProductTelemetry(ABC):
    """Base interface for capturing telemetry data (no-op implementation)."""

    @abstractmethod
    def capture(self, event: BaseTelemetryEvent) -> None:
        """Capture a telemetry event (no-op).

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture (ignored)

        """


@singleton
class ProductTelemetry(BaseProductTelemetry):
    """No-op telemetry service (telemetry removed)."""

    def __init__(self) -> None:
        """Initialize the no-op telemetry service."""

    def capture(self, event: BaseTelemetryEvent) -> None:
        """No-op capture method - does nothing.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture (ignored)

        """
