"""Unit tests for the no-op telemetry service module."""

from portia.telemetry.telemetry_service import ProductTelemetry
from portia.telemetry.views import BaseTelemetryEvent


class TelemetryEvent(BaseTelemetryEvent):
    """Test implementation of BaseTelemetryEvent for testing purposes."""

    def __init__(self, name: str) -> None:
        """Initialize the test telemetry event.

        Args:
            name: The name of the event.

        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the event name.

        Returns:
            The name of the event.

        """
        return self._name


def test_telemetry_init() -> None:
    """Test that ProductTelemetry initializes correctly as no-op."""
    telemetry = ProductTelemetry()
    assert telemetry is not None


def test_telemetry_capture_no_op() -> None:
    """Test that telemetry capture method is a no-op and doesn't raise exceptions."""
    telemetry = ProductTelemetry()
    event = TelemetryEvent("test_event")

    # Should not raise any exceptions
    telemetry.capture(event)

    # Should be able to call multiple times without issues
    telemetry.capture(event)
    telemetry.capture(event)


def test_telemetry_properties_empty() -> None:
    """Test that telemetry event properties return empty dict."""
    event = TelemetryEvent("test_event")
    assert event.properties == {}
