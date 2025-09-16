"""Unit tests for the telemetry service module."""

from typing import Any

import pytest

from portia.telemetry.telemetry_service import ProductTelemetry
from portia.telemetry.views import BaseTelemetryEvent


class TelemetryEvent(BaseTelemetryEvent):
    """Test implementation of BaseTelemetryEvent for testing purposes."""

    def __init__(self, name: str, properties: dict) -> None:
        """Initialize the test telemetry event.

        Args:
            name: The name of the event.
            properties: The properties of the event.

        """
        self._name = name
        self._properties = properties

    @property
    def name(self) -> str:
        """Get the event name.

        Returns:
            The name of the event.

        """
        return self._name

    @property
    def properties(self) -> dict:
        """Get the event properties.

        Returns:
            The properties of the event.

        """
        return self._properties


@pytest.fixture
def telemetry() -> Any:  # noqa: ANN401
    """Create a fresh ProductTelemetry instance for each test.

    Returns:
        A new ProductTelemetry instance.

    """
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
    yield ProductTelemetry()
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue


def test_telemetry_initialization() -> None:
    """Test that telemetry initializes without errors."""
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
    telemetry = ProductTelemetry()
    assert telemetry is not None


def test_capture_no_op(telemetry: ProductTelemetry) -> None:  # type: ignore reportGeneralTypeIssues
    """Test that capture method is a no-op and doesn't raise exceptions.

    Args:
        telemetry: The ProductTelemetry instance to test.

    """
    event = TelemetryEvent("test_event", {"key": "value"})
    # Should not raise any exceptions and should return None
    result = telemetry.capture(event)
    assert result is None


def test_capture_multiple_events(telemetry: ProductTelemetry) -> None:  # type: ignore reportGeneralTypeIssues
    """Test that multiple capture calls work without issues.

    Args:
        telemetry: The ProductTelemetry instance to test.

    """
    events = [
        TelemetryEvent("event1", {"key1": "value1"}),
        TelemetryEvent("event2", {"key2": "value2"}),
        TelemetryEvent("event3", {}),
    ]

    # Should not raise any exceptions
    for event in events:
        telemetry.capture(event)


def test_singleton_behavior() -> None:
    """Test that ProductTelemetry behaves as a singleton."""
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
    telemetry1 = ProductTelemetry()
    telemetry2 = ProductTelemetry()
    assert telemetry1 is telemetry2
