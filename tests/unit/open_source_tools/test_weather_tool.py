"""Weather tool tests."""

from unittest.mock import patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.weather import WeatherTool
from tests.utils import get_test_tool_context


def test_weather_tool_missing_api_key() -> None:
    """Test that WeatherTool raises ToolHardError if API key is missing."""
    tool = WeatherTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError):
            tool.run(ctx, "paris")


def test_weather_tool_successful_response(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool successfully processes a valid response."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {"main": {"temp": 10}, "weather": [{"description": "sunny"}]}

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API response using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=paris&appid=mock-api-key&units=metric",
            json=mock_response,
            status_code=200,
        )

        ctx = get_test_tool_context()
        result = tool.run(ctx, "paris")
        assert result == "The current weather in paris is sunny with a temperature of 10°C."


def test_weather_tool_no_answer_in_response(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool raises ToolSoftError if no answer is found in the response."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {"no_answer": "No relevant information found."}

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API response using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=Paris&appid=mock-api-key&units=metric",
            json=mock_response,
            status_code=200,
        )

        ctx = get_test_tool_context()
        with pytest.raises(ToolSoftError, match="No data found for: Paris"):
            tool.run(ctx, "Paris")


def test_weather_tool_no_main_answer_in_response(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool raises ToolSoftError if no answer is found in the response."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {
        "no_answer": "No relevant information found.",
        "weather": [{"description": "sunny"}],
    }

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API response using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=Paris&appid=mock-api-key&units=metric",
            json=mock_response,
            status_code=200,
        )

        ctx = get_test_tool_context()
        with pytest.raises(ToolSoftError, match="No main data found for city: Paris"):
            tool.run(ctx, "Paris")


def test_weather_tool_http_error(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool handles HTTP errors correctly."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API to raise an exception using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=Paris&appid=mock-api-key&units=metric",
            status_code=500,
        )

        ctx = get_test_tool_context()
        with pytest.raises(httpx.HTTPStatusError):
            tool.run(ctx, "Paris")


# Async tests using httpx_mock
@pytest.mark.asyncio
async def test_weather_tool_async_missing_api_key() -> None:
    """Test that WeatherTool raises ToolHardError if API key is missing (async)."""
    tool = WeatherTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError):
            await tool.arun(ctx, "paris")


@pytest.mark.asyncio
async def test_weather_tool_async_successful_response(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool successfully processes a valid response (async)."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {"main": {"temp": 10}, "weather": [{"description": "sunny"}]}

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API response using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=paris&appid=mock-api-key&units=metric",
            json=mock_response,
            status_code=200,
        )

        ctx = get_test_tool_context()
        result = await tool.arun(ctx, "paris")
        assert result == "The current weather in paris is sunny with a temperature of 10°C."


@pytest.mark.asyncio
async def test_weather_tool_async_no_answer_in_response(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool raises ToolSoftError if no answer is found in the response (async)."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {"no_answer": "No relevant information found."}

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API response using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=Paris&appid=mock-api-key&units=metric",
            json=mock_response,
            status_code=200,
        )

        ctx = get_test_tool_context()
        with pytest.raises(ToolSoftError, match="No data found for: Paris"):
            await tool.arun(ctx, "Paris")


@pytest.mark.asyncio
async def test_weather_tool_async_no_main_answer_in_response(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool raises ToolSoftError if no answer is found in the response (async)."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"
    mock_response = {
        "no_answer": "No relevant information found.",
        "weather": [{"description": "sunny"}],
    }

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API response using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=Paris&appid=mock-api-key&units=metric",
            json=mock_response,
            status_code=200,
        )

        ctx = get_test_tool_context()
        with pytest.raises(ToolSoftError, match="No main data found for city: Paris"):
            await tool.arun(ctx, "Paris")


@pytest.mark.asyncio
async def test_weather_tool_async_http_error(httpx_mock: HTTPXMock) -> None:
    """Test that WeatherTool handles HTTP errors correctly (async)."""
    tool = WeatherTool()
    mock_api_key = "mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        # Mock the API to raise an exception using httpx_mock
        httpx_mock.add_response(
            url="http://api.openweathermap.org/data/2.5/weather?q=Paris&appid=mock-api-key&units=metric",
            status_code=500,
        )

        ctx = get_test_tool_context()
        with pytest.raises(httpx.HTTPStatusError):
            await tool.arun(ctx, "Paris")
