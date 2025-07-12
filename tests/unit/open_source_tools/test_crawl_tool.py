"""Crawl tool tests."""

from unittest.mock import Mock, patch

import pytest

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.crawl_tool import CrawlTool
from tests.utils import get_test_tool_context


def test_crawl_tool_missing_api_key() -> None:
    """Test that CrawlTool raises ToolHardError if API key is missing."""
    tool = CrawlTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError):
            tool.run(ctx, "https://docs.tavily.com")


def test_crawl_tool_successful_response() -> None:
    """Test that CrawlTool successfully processes a valid response."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "docs.tavily.com",
        "results": [
            {
                "url": "https://docs.tavily.com/welcome",
                "raw_content": "Welcome - Tavily Docs content...",
                "favicon": "https://mintlify.s3-us-west-1.amazonaws.com/tavilyai/_generated/favicon/apple-touch-icon.png?v=3",
            },
            {
                "url": "https://docs.tavily.com/sdk/python/quick-start",
                "raw_content": "Python SDK quickstart content...",
                "favicon": "https://mintlify.s3-us-west-1.amazonaws.com/tavilyai/_generated/favicon/apple-touch-icon.png?v=3",
            },
        ],
        "response_time": 1.23,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(
                ctx, "https://docs.tavily.com", instructions="Find all pages on the Python SDK"
            )
            assert "Crawled 2 pages:" in result
            assert "https://docs.tavily.com/welcome" in result
            assert "https://docs.tavily.com/sdk/python/quick-start" in result


def test_crawl_tool_with_advanced_options() -> None:
    """Test that CrawlTool works with advanced crawl options."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/docs/api",
                "raw_content": "API documentation content",
                "favicon": "https://example.com/favicon.ico",
            }
        ],
        "response_time": 0.8,
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
                instructions="Find API documentation",
                select_paths=["/docs/.*", "/api/.*"],
                exclude_paths=["/private/.*"],
                allow_external=True,
            )
            assert "Crawled 1 pages:" in result
            assert "https://example.com/docs/api" in result

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["max_depth"] == 2
            assert payload["max_breadth"] == 10
            assert payload["limit"] == 25
            assert payload["instructions"] == "Find API documentation"
            assert payload["select_paths"] == ["/docs/.*", "/api/.*"]
            assert payload["exclude_paths"] == ["/private/.*"]
            assert payload["allow_external"] is True
            assert "include_images" not in payload
            assert "categories" not in payload
            assert "extract_depth" not in payload
            assert "format" not in payload
            assert "include_favicon" not in payload


def test_crawl_tool_default_parameters() -> None:
    """Test that CrawlTool uses correct default parameters."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/page1",
                "raw_content": "Default crawl content",
                "favicon": "",
            }
        ],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "https://example.com")
            assert "Crawled 1 pages:" in result
            assert "https://example.com/page1" in result

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert "max_depth" not in payload
            assert "max_breadth" not in payload
            assert "limit" not in payload
            assert "allow_external" not in payload
            assert "instructions" not in payload
            assert "select_paths" not in payload
            assert "select_domains" not in payload
            assert "exclude_paths" not in payload
            assert "exclude_domains" not in payload


def test_crawl_tool_optional_parameters_only_when_provided() -> None:
    """Test that optional parameters are only included when provided."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"base_url": "example.com", "results": []}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(
                ctx, "https://example.com", instructions="Find docs", select_paths=["/docs/.*"]
            )

            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert "instructions" in payload
            assert "select_paths" in payload
            assert "select_domains" not in payload
            assert "exclude_paths" not in payload


def test_crawl_tool_no_results_in_response() -> None:
    """Test that CrawlTool raises ToolSoftError if no results are found in the response."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"error": "No pages could be crawled"}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="Failed to crawl website:.*"):
                tool.run(ctx, "https://example.com")


def test_crawl_tool_http_error() -> None:
    """Test that CrawlTool handles HTTP errors correctly."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"

    with (
        patch("os.getenv", return_value=mock_api_key),
        patch("httpx.post", side_effect=Exception("HTTP Error")),
    ):
        ctx = get_test_tool_context()
        with pytest.raises(ToolSoftError, match="Crawl request failed:.*"):
            tool.run(ctx, "https://example.com")


def test_crawl_tool_authorization_header() -> None:
    """Test that CrawlTool uses correct authorization header."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/",
                "raw_content": "Example content",
            }
        ],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(ctx, "https://example.com")

            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == f"Bearer {mock_api_key}"
            assert headers["Content-Type"] == "application/json"


def test_crawl_tool_api_endpoint() -> None:
    """Test that CrawlTool calls the correct API endpoint."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {"base_url": "example.com", "results": []}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            tool.run(ctx, "https://example.com")

            call_args = mock_post.call_args
            assert call_args[0][0] == "https://api.tavily.com/crawl"


def test_crawl_tool_timeout_error() -> None:
    """Test that CrawlTool handles timeout errors correctly."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("Request timed out")):
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="Crawl request timed out"):
                tool.run(ctx, "https://example.com")


def test_crawl_tool_http_status_error() -> None:
    """Test that CrawlTool handles HTTP status errors correctly."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        import httpx

        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"error": "Invalid parameters"}
        mock_response.text = '{"error": "Invalid parameters"}'

        with patch("httpx.post") as mock_post:
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "422 Unprocessable Entity", request=None, response=mock_response
            )

            with pytest.raises(ToolSoftError, match="Crawl API error - HTTP 422:.*"):
                tool.run(ctx, "https://example.com")


def test_crawl_tool_exclude_paths_parameter() -> None:
    """Test that CrawlTool correctly includes exclude_paths parameter when provided."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/public/page1",
                "raw_content": "Public content",
            }
        ],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(
                ctx,
                "https://example.com",
                exclude_paths=["/private/.*", "/admin/.*"],
            )
            assert "Crawled 1 pages:" in result
            assert "https://example.com/public/page1" in result

            # Verify the exclude_paths parameter is included in the payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert "exclude_paths" in payload
            assert payload["exclude_paths"] == ["/private/.*", "/admin/.*"]


def test_crawl_tool_exclude_domains_parameter() -> None:
    """Test that CrawlTool correctly includes exclude_domains parameter when provided."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/page1",
                "raw_content": "Main domain content",
            }
        ],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(
                ctx,
                "https://example.com",
                exclude_domains=["^private\\.example\\.com$", "^admin\\.example\\.com$"],
            )
            assert "Crawled 1 pages:" in result
            assert "https://example.com/page1" in result

            # Verify the exclude_domains parameter is included in the payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert "exclude_domains" in payload
            assert payload["exclude_domains"] == [
                "^private\\.example\\.com$",
                "^admin\\.example\\.com$",
            ]


def test_crawl_tool_all_optional_parameters() -> None:
    """Test that CrawlTool correctly handles all optional parameters."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"
    mock_response = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/docs/api",
                "raw_content": "API documentation content",
            }
        ],
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(
                ctx,
                "https://example.com",
                instructions="Find API docs",
                max_depth=2,
                max_breadth=10,
                limit=25,
                select_paths=["/docs/.*", "/api/.*"],
                select_domains=["^docs\\.example\\.com$"],
                exclude_paths=["/private/.*"],
                exclude_domains=["^private\\.example\\.com$"],
                allow_external=True,
            )
            assert "Crawled 1 pages:" in result
            assert "https://example.com/docs/api" in result

            # Verify all parameters are included in the payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["instructions"] == "Find API docs"
            assert payload["max_depth"] == 2
            assert payload["max_breadth"] == 10
            assert payload["limit"] == 25
            assert payload["select_paths"] == ["/docs/.*", "/api/.*"]
            assert payload["select_domains"] == ["^docs\\.example\\.com$"]
            assert payload["exclude_paths"] == ["/private/.*"]
            assert payload["exclude_domains"] == ["^private\\.example\\.com$"]
            assert payload["allow_external"] is True


def test_crawl_tool_http_status_error_with_json_response() -> None:
    """Test that CrawlTool handles HTTP status errors with JSON response body correctly."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        import httpx

        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "error": "Invalid parameters",
            "details": "URL is required",
        }
        mock_response.text = '{"error": "Invalid parameters", "details": "URL is required"}'

        with (
            patch("os.getenv", return_value=mock_api_key),
            patch("httpx.post", side_effect=httpx.HTTPStatusError(
                message="Unprocessable Entity",
                request=Mock(),
                response=mock_response,
            )),
        ):
            ctx = get_test_tool_context()

            with pytest.raises(
                ToolSoftError,
                match="Crawl API error - HTTP 422:.*Invalid parameters.*",
            ):
                tool.run(ctx, "https://example.com")


def test_crawl_tool_http_status_error_with_invalid_json() -> None:
    """Test that CrawlTool handles HTTP status errors with invalid JSON response body correctly."""
    tool = CrawlTool()
    mock_api_key = "tvly-mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        import httpx

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Internal Server Error"

        with patch("httpx.post") as mock_post:
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error", request=None, response=mock_response
            )

            with pytest.raises(
                ToolSoftError,
                match="Crawl API error - HTTP 500: Internal Server Error",
            ):
                tool.run(ctx, "https://example.com")
