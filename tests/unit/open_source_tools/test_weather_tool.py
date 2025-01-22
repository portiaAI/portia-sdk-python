"""Weather tool tests."""

from unittest.mock import Mock, patch

import pytest

from portia.context import empty_context
from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.weather import WeatherTool


def test_weather_tool_missing_api_key() -> None:
    """Test that WeatherTool raises ToolHardError if API key is missing."""
    tool = WeatherTool()
    with patch("os.getenv", return_value=""):
        ctx = empty_context()
        with pytest.raises(ToolHardError):
            tool.run(ctx, "paris")


def test_weather_tool_successful_response() -> None:
    """Test that WeatherTool successfully processes a valid response."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {"main": {"temp": 10}, "weather": [{"description": "sunny"}]}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = empty_context()
        with patch("httpx.get") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "paris")
            assert result == "The current weather in paris is sunny with a temperature of 10°C."


def test_weather_tool_no_answer_in_response() -> None:
    """Test that WeatherTool raises ToolSoftError if no answer is found in the response."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {"no_answer": "No relevant information found."}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = empty_context()
        with patch("httpx.get") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="No data found for: Paris"):
                tool.run(ctx, "Paris")


def test_weather_tool_no_main_answer_in_response() -> None:
    """Test that WeatherTool raises ToolSoftError if no answer is found in the response."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {
        "no_answer": "No relevant information found.",
        "weather": [{"description": "sunny"}],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = empty_context()
        with patch("httpx.get") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="No main data found for city: Paris"):
                tool.run(ctx, "Paris")


def test_weather_tool_http_error() -> None:
    """Test that WeatherTool handles HTTP errors correctly."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):  # noqa: SIM117
        with patch("httpx.get", side_effect=Exception("HTTP Error")):
            ctx = empty_context()
            with pytest.raises(Exception, match="HTTP Error"):
                tool.run(ctx, "Paris")
