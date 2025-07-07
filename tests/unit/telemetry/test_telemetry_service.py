"""Unit tests for the telemetry service module."""

import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from posthog import Posthog

from portia.telemetry.telemetry_service import (
    ProductTelemetry,
    get_project_id_key,
    xdg_cache_home,
)
from portia.telemetry.views import BaseTelemetryEvent


class TestTelemetryEvent(BaseTelemetryEvent):
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


def test_xdg_cache_home_default() -> None:
    """Test xdg_cache_home function with default environment."""
    with patch.dict(os.environ, {}, clear=True):
        assert xdg_cache_home() == Path.home() / ".portia"


def test_xdg_cache_home_custom() -> None:
    """Test xdg_cache_home function with custom XDG_CACHE_HOME."""
    custom_path = "/custom/cache/path"
    with patch.dict(os.environ, {"XDG_CACHE_HOME": custom_path}, clear=True):
        assert xdg_cache_home() == Path(custom_path)


def test_get_project_id_key_localhost() -> None:
    """Test get_project_id_key function with localhost endpoint."""
    with patch.dict(os.environ, {"PORTIA_API_ENDPOINT": "http://localhost:8000"}, clear=True):
        assert get_project_id_key() == "phc_QHjx4dKKNAqmLS1U64kIXo4NlYOGIFDgB1qYxw3wh1W"


def test_get_project_id_key_dev() -> None:
    """Test get_project_id_key function with dev endpoint."""
    with patch.dict(os.environ, {"PORTIA_API_ENDPOINT": "https://dev.portia.com"}, clear=True):
        assert get_project_id_key() == "phc_gkmBfAtjABu5dDAX9KX61iAF10Wyze4FGPrT3g7mcKo"


def test_get_project_id_key_default() -> None:
    """Test get_project_id_key function with default endpoint."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_project_id_key() == "phc_fGJERhs0sljicW5IFBzJZoenOb0jtsIcAghCZHw97V1"


class TestProductTelemetry:
    """Test suite for ProductTelemetry class."""

    @pytest.fixture
    def telemetry(self) -> Any:  # noqa: ANN401
        """Create a fresh ProductTelemetry instance for each test.

        Returns:
            A new ProductTelemetry instance.

        """
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
        yield ProductTelemetry()
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Mock logger for testing."""
        logger = MagicMock()

        logger.level = "DEBUG"
        return logger

    def test_init_telemetry_disabled(self, mock_logger: MagicMock) -> None:
        """Test initialization with telemetry disabled."""
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
        with (
            patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "false"}, clear=True),
            patch("portia.telemetry.telemetry_service.logger", mock_logger),
        ):
            telemetry = ProductTelemetry()
            mock_logger.debug.assert_called_once_with("Telemetry disabled")
            assert telemetry._posthog_client is None
            assert logging.getLogger("posthog").disabled

    def test_init_telemetry_enabled(self, mock_logger: MagicMock) -> None:
        """Test initialization with telemetry enabled."""
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
        with (
            patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "true"}, clear=True),
            patch("portia.telemetry.telemetry_service.logger", mock_logger),
        ):
            telemetry = ProductTelemetry()
            mock_logger.info.assert_called_once()
            assert "Portia anonymized telemetry enabled" in mock_logger.info.call_args[0][0]
            assert isinstance(telemetry._posthog_client, Posthog)

    def test_capture_when_disabled(self, mock_logger: MagicMock) -> None:
        """Test event capture when telemetry is disabled."""
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
        with (
            patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "false"}, clear=True),
            patch("portia.telemetry.telemetry_service.logger", mock_logger),
        ):
            telemetry = ProductTelemetry()
            event = TestTelemetryEvent("test_event", {})
            # Should not raise any exceptions
            telemetry.capture(event)
            mock_logger.debug.assert_called_once_with("Telemetry disabled")

    def test_capture_when_enabled(self, mock_logger: MagicMock) -> None:
        """Test event capture when telemetry is enabled."""
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
        with (
            patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "true"}, clear=True),
            patch("portia.telemetry.telemetry_service.logger", mock_logger),
        ):
            telemetry = ProductTelemetry()
            mock_client = MagicMock()
            telemetry._posthog_client = mock_client

            event = TestTelemetryEvent("test_event", {"key": "value"})
            telemetry.capture(event)

            mock_logger.debug.assert_called_with("Telemetry event: test_event {'key': 'value'}")
            mock_client.capture.assert_called_once()
            args = mock_client.capture.call_args[0]
            assert args[0] == "test_event"

            kwargs = mock_client.capture.call_args[1]
            assert kwargs["properties"]["key"] == "value"
            assert kwargs["properties"]["process_person_profile"] is True

    def test_capture_when_enabled_with_exception(self, mock_logger: MagicMock) -> None:
        """Test event capture when telemetry is enabled and PostHog client raises an exception."""
        ProductTelemetry.reset()  # type: ignore reportAccessAttributeIssue
        with (
            patch.dict(os.environ, {"ANONYMIZED_TELEMETRY": "true"}, clear=True),
            patch("portia.telemetry.telemetry_service.logger", mock_logger),
        ):
            telemetry = ProductTelemetry()
            mock_client = MagicMock()
            mock_client.capture.side_effect = Exception("PostHog API error")
            telemetry._posthog_client = mock_client

            event = TestTelemetryEvent("test_event", {"key": "value"})
            # Should not raise the exception
            telemetry.capture(event)

            mock_logger.debug.assert_called_with("Telemetry event: test_event {'key': 'value'}")
            mock_client.capture.assert_called_once()
            mock_logger.exception.assert_called_once()
            assert "Failed to send telemetry event" in mock_logger.exception.call_args[0][0]

    def test_user_id_generation(self, telemetry: ProductTelemetry, tmp_path: Path) -> None:  # type: ignore reportGeneralTypeIssues
        """Test user ID generation and persistence.

        Args:
            telemetry: The ProductTelemetry instance to test.
            tmp_path: Temporary directory path for testing.

        """
        with patch.object(telemetry, "USER_ID_PATH", str(tmp_path / "user_id")):
            # First call should generate a new ID
            user_id1 = telemetry.user_id
            assert user_id1 != "UNKNOWN_USER_ID"

            # Second call should return the same ID
            user_id2 = telemetry.user_id
            assert user_id1 == user_id2

            telemetry._curr_user_id = None
            # Third call after reset should return the same ID
            user_id3 = telemetry.user_id
            assert user_id1 == user_id3

    def test_user_id_error_handling(self, telemetry: ProductTelemetry) -> None:  # type: ignore reportGeneralTypeIssues
        """Test user ID error handling with invalid path.

        Args:
            telemetry: The ProductTelemetry instance to test.

        """
        with patch.object(telemetry, "USER_ID_PATH", "/invalid/path/user_id"):
            assert telemetry.user_id == "UNKNOWN_USER_ID"
