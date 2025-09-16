"""Telemetry service for capturing anonymized usage data."""

from abc import ABC, abstractmethod

from portia.common import singleton
from portia.telemetry.views import BaseTelemetryEvent


class BaseProductTelemetry(ABC):
    """Base interface for capturing anonymized telemetry data.

    This class handles the collection and transmission of anonymized usage data to PostHog.
    Telemetry can be disabled by setting the environment variable `ANONYMIZED_TELEMETRY=False`.

    """

    @abstractmethod
    def capture(self, event: BaseTelemetryEvent) -> None:
        """Capture and send a telemetry event.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture

        """


@singleton
class ProductTelemetry(BaseProductTelemetry):
    """No-op telemetry service.

    This implementation provides a no-op telemetry service that maintains interface
    compatibility but does not collect or transmit any data.
    """

    def capture(self, event: BaseTelemetryEvent) -> None:
        """No-op capture method.

        Args:
            event (BaseTelemetryEvent): The telemetry event (ignored)

        """
        # No-op implementation - telemetry has been disabled
