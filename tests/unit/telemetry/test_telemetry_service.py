"""Unit tests for the telemetry service module."""

from unittest.mock import MagicMock, patch

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
def mock_logger() -> MagicMock:
    """Mock logger for testing."""
    return MagicMock()


def test_product_telemetry_init(mock_logger: MagicMock) -> None:
    """Test ProductTelemetry initialization."""
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
    with patch("portia.telemetry.telemetry_service.logger", mock_logger):
        telemetry = ProductTelemetry()
        mock_logger.debug.assert_called_once_with("Telemetry disabled - using no-op implementation")
        assert telemetry is not None


def test_product_telemetry_capture(mock_logger: MagicMock) -> None:
    """Test ProductTelemetry capture method is no-op."""
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
    with patch("portia.telemetry.telemetry_service.logger", mock_logger):
        telemetry = ProductTelemetry()
        event = TelemetryEvent("test_event", {"key": "value"})

        # Should not raise any exceptions
        telemetry.capture(event)

        # Verify initialization debug message was called
        mock_logger.debug.assert_called_once_with("Telemetry disabled - using no-op implementation")


def test_singleton_behavior() -> None:
    """Test that ProductTelemetry follows singleton pattern."""
    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue

    telemetry1 = ProductTelemetry()
    telemetry2 = ProductTelemetry()

    assert telemetry1 is telemetry2

    ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
