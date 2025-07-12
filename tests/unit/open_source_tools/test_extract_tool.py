"""Extract tool tests."""

from unittest.mock import Mock, patch

import pytest

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.extract_tool import ExtractTool
from tests.utils import get_test_tool_context


def test_extract_tool_missing_api_key() -> None:
    """Test that ExtractTool raises ToolHardError if API key is missing."""
    tool = ExtractTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError):
            tool.run(ctx, ["https://example.com"])


def test_extract_tool_successful_response() -> None:
    """Test that ExtractTool successfully processes a valid response."""
    tool = ExtractTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "This is the content of the webpage",
                "images": ["https://example.com/image1.jpg"],
                "favicon": "https://example.com/favicon.ico",
            }
        ],
        "failed_results": [],
        "response_time": 0.5,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, ["https://example.com"])
            assert result == mock_response["results"]


def test_extract_tool_multiple_urls() -> None:
    """Test that ExtractTool successfully processes multiple URLs."""
    tool = ExtractTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "results": [
            {
                "url": "https://example1.com",
                "raw_content": "Content from first URL",
                "images": [],
                "favicon": "https://example1.com/favicon.ico",
            },
            {
                "url": "https://example2.com",
                "raw_content": "Content from second URL",
                "images": ["https://example2.com/image.jpg"],
                "favicon": "https://example2.com/favicon.ico",
            },
        ],
        "failed_results": [],
        "response_time": 1.2,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, ["https://example1.com", "https://example2.com"])
            assert result == mock_response["results"]


def test_extract_tool_with_custom_options() -> None:
    """Test that ExtractTool works with custom extraction options."""
    tool = ExtractTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Advanced extracted content",
                "images": [],
                "favicon": "",
            }
        ],
        "failed_results": [],
        "response_time": 0.8,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(
                ctx,
                ["https://example.com"],
                include_images=False,
                include_favicon=False,
                extract_depth="advanced",
                format="text",
            )
            assert result == mock_response["results"]

            # Verify the payload was constructed correctly
            call_args = mock_post.call_args
            assert call_args[1]["json"]["include_images"] is False
            assert call_args[1]["json"]["include_favicon"] is False
            assert call_args[1]["json"]["extract_depth"] == "advanced"
            assert call_args[1]["json"]["format"] == "text"


def test_extract_tool_no_results_in_response() -> None:
    """Test that ExtractTool raises ToolSoftError if no results are found in the response."""
    tool = ExtractTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"error": "No content could be extracted"}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="Failed to extract content:.*"):
                tool.run(ctx, ["https://example.com"])


def test_extract_tool_http_error() -> None:
    """Test that ExtractTool handles HTTP errors correctly."""
    tool = ExtractTool()
    mock_api_key = "tvly-mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):  # noqa: SIM117
        with patch("httpx.post", side_effect=Exception("HTTP Error")):
            ctx = get_test_tool_context()
            with pytest.raises(Exception, match="HTTP Error"):
                tool.run(ctx, ["https://example.com"])


def test_extract_tool_authorization_header() -> None:
    """Test that ExtractTool uses correct authorization header."""
    tool = ExtractTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Test content",
                "images": [],
                "favicon": "",
            }
        ]
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(ctx, ["https://example.com"])

            # Verify the authorization header is set correctly
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == f"Bearer {mock_api_key}"
            assert headers["Content-Type"] == "application/json"
