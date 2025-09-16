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


def test_product_telemetry_no_op_capture() -> None:
    """Test that ProductTelemetry capture method is a no-op."""
    telemetry = ProductTelemetry()
    event = TelemetryEvent("test_event", {"key": "value"})

    # Should not raise any exceptions and should be a no-op
    telemetry.capture(event)


def test_product_telemetry_initialization() -> None:
    """Test ProductTelemetry initialization."""
    telemetry = ProductTelemetry()

    # Should be able to instantiate without issues
    assert telemetry is not None
    assert callable(telemetry.capture)
