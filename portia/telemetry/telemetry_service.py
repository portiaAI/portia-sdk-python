"""Telemetry service for capturing anonymized usage data."""

import logging
from abc import ABC, abstractmethod

from portia.common import singleton
from portia.telemetry.views import BaseTelemetryEvent

logger = logging.getLogger(__name__)


class BaseProductTelemetry(ABC):
    """Base interface for capturing anonymized telemetry data.

    This abstract base class defines the interface for telemetry data collection.
    """

    @abstractmethod
    def capture(self, event: BaseTelemetryEvent) -> None:
        """Capture and send a telemetry event.

        Args:
            event (BaseTelemetryEvent): The telemetry event to capture

        """


@singleton
class ProductTelemetry(BaseProductTelemetry):
    """No-op service for telemetry interface compatibility.

    This class provides a no-op implementation of telemetry to maintain interface
    compatibility while disabling all telemetry collection and transmission.
    """

    def __init__(self) -> None:
        """Initialize the no-op telemetry service."""
        logger.debug("Telemetry disabled - using no-op implementation")

    def capture(self, event: BaseTelemetryEvent) -> None:
        """No-op capture method.

        Args:
            event (BaseTelemetryEvent): The telemetry event (ignored)

        """
        # No-op implementation - does nothing
