"""Map tool tests."""

from unittest.mock import Mock, patch

import pytest

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.map_tool import MapTool
from tests.utils import get_test_tool_context


def test_map_tool_missing_api_key() -> None:
    """Test that MapTool raises ToolHardError if API key is missing."""
    tool = MapTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError):
            tool.run(ctx, "https://docs.tavily.com")


def test_map_tool_successful_response() -> None:
    """Test that MapTool successfully processes a valid response."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "docs.tavily.com",
        "results": [
            "https://docs.tavily.com/welcome",
            "https://docs.tavily.com/documentation/api-credits",
            "https://docs.tavily.com/documentation/about",
        ],
        "response_time": 1.23,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "https://docs.tavily.com")
            assert result == mock_response["results"]


def test_map_tool_with_advanced_options() -> None:
    """Test that MapTool works with advanced mapping options."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            "https://example.com/docs/api",
            "https://example.com/docs/guides",
            "https://example.com/docs/tutorials",
        ],
        "response_time": 2.45,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(
                ctx,
                "https://example.com",
                max_depth=2,
                max_breadth=10,
                limit=25,
                instructions="Find documentation pages",
                select_paths=["/docs/.*", "/api/.*"],
                exclude_paths=["/private/.*"],
                allow_external=True,
                categories=["Documentation", "API"],
            )
            assert result == mock_response["results"]

            # Verify the payload was constructed correctly
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["max_depth"] == 2
            assert payload["max_breadth"] == 10
            assert payload["limit"] == 25
            assert payload["instructions"] == "Find documentation pages"
            assert payload["select_paths"] == ["/docs/.*", "/api/.*"]
            assert payload["exclude_paths"] == ["/private/.*"]
            assert payload["allow_external"] is True
            assert payload["categories"] == ["Documentation", "API"]


def test_map_tool_default_parameters() -> None:
    """Test that MapTool uses correct default parameters."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/contact",
        ],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "https://example.com")
            assert result == mock_response["results"]

            # Verify default parameters
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["max_depth"] == 1
            assert payload["max_breadth"] == 20
            assert payload["limit"] == 50
            assert payload["allow_external"] is False
            # Optional parameters should not be in payload when None
            assert "instructions" not in payload
            assert "select_paths" not in payload
            assert "select_domains" not in payload
            assert "exclude_paths" not in payload
            assert "exclude_domains" not in payload
            assert "categories" not in payload


def test_map_tool_optional_parameters_only_when_provided() -> None:
    """Test that optional parameters are only included when provided."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"base_url": "example.com", "results": []}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            # Test with only some optional parameters
            tool.run(
                ctx, "https://example.com", instructions="Find docs", select_paths=["/docs/.*"]
            )

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert "instructions" in payload
            assert "select_paths" in payload
            # These should not be present
            assert "select_domains" not in payload
            assert "exclude_paths" not in payload
            assert "categories" not in payload


def test_map_tool_complex_website() -> None:
    """Test that MapTool successfully maps a more complex website."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/products",
            "https://example.com/blog",
            "https://example.com/contact",
            "https://example.com/blog/post-1",
            "https://example.com/blog/post-2",
            "https://example.com/products/product-a",
            "https://example.com/products/product-b",
        ],
        "response_time": 2.45,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "https://example.com", max_depth=2, limit=100)
            assert result == mock_response["results"]
            assert len(result) == 9  # Verify we got all URLs


def test_map_tool_single_page_site() -> None:
    """Test that MapTool handles single-page websites correctly."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "simple.com",
        "results": ["https://simple.com/"],
        "response_time": 0.5,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "https://simple.com")
            assert result == mock_response["results"]
            assert len(result) == 1


def test_map_tool_no_results_in_response() -> None:
    """Test that MapTool raises ToolSoftError if no results are found in the response."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"error": "No pages could be mapped"}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="Failed to map website:.*"):
                tool.run(ctx, "https://example.com")


def test_map_tool_http_error() -> None:
    """Test that MapTool handles HTTP errors correctly."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):  # noqa: SIM117
        with patch("httpx.post", side_effect=Exception("HTTP Error")):
            ctx = get_test_tool_context()
            with pytest.raises(Exception, match="HTTP Error"):
                tool.run(ctx, "https://example.com")


def test_map_tool_authorization_header() -> None:
    """Test that MapTool uses correct authorization header."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"base_url": "example.com", "results": ["https://example.com/"]}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(ctx, "https://example.com")

            # Verify the authorization header is set correctly
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == f"Bearer {mock_api_key}"
            assert headers["Content-Type"] == "application/json"


def test_map_tool_api_endpoint() -> None:
    """Test that MapTool calls the correct API endpoint."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"base_url": "example.com", "results": []}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(ctx, "https://example.com")

            # Verify the correct endpoint is called
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://api.tavily.com/map"


def test_map_tool_payload_structure() -> None:
    """Test that MapTool sends correct payload structure with defaults."""
    tool = MapTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"base_url": "example.com", "results": ["https://example.com/"]}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(ctx, "https://example.com")

            # Verify the payload structure with default values
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["max_depth"] == 1
            assert payload["max_breadth"] == 20
            assert payload["limit"] == 50
            assert payload["allow_external"] is False
            # Optional parameters should not be present when None
            assert "instructions" not in payload
            assert "select_paths" not in payload
