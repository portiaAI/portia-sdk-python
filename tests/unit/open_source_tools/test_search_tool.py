"""Search tool tests."""

from unittest.mock import Mock, patch

import pytest

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.search_tool import SearchTool
from tests.utils import get_test_tool_context


def test_search_tool_missing_api_key() -> None:
    """Test that SearchTool raises ToolHardError if API key is missing."""
    tool = SearchTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError):
            tool.run(ctx, "What is the capital of France?")


def test_search_tool_successful_response() -> None:
    """Test that SearchTool successfully processes a valid response."""
    tool = SearchTool()
    mock_api_key = "mock-api-key"
    mock_response = {
        "query": "What is the capital of France?",
        "follow_up_questions": "",
        "answer": "The capital of France is Paris.",
        "images": [],
        "results": ["result1", "result2", "result3", "result4", "result5"],
        "response_time": 2.43,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "What is the capital of France?")
            assert result == ["result1", "result2", "result3"]

def test_search_tool_fewer_results_than_max() -> None:
    """Test that SearchTool successfully processes a valid response."""
    tool = SearchTool()
    mock_api_key = "mock-api-key"
    mock_response = {
        "query": "What is the capital of France?",
        "follow_up_questions": "",
        "answer": "The capital of France is Paris.",
        "images": [],
        "results": ["result1", "result2"],
        "response_time": 2.43,
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "What is the capital of France?")
            assert result == ["result1", "result2"]

def test_search_tool_no_answer_in_response() -> None:
    """Test that SearchTool raises ToolSoftError if no answer is found in the response."""
    tool = SearchTool()
    mock_api_key = "mock-api-key"
    mock_response = {"no_answer": "No relevant information found."}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="Failed to get answer to search:.*"):
                tool.run(ctx, "What is the capital of France?")


def test_search_tool_http_error() -> None:
    """Test that SearchTool handles HTTP errors correctly."""
    tool = SearchTool()
    mock_api_key = "mock-api-key"

    with patch("os.getenv", return_value=mock_api_key):  # noqa: SIM117
        with patch("httpx.post", side_effect=Exception("HTTP Error")):
            ctx = get_test_tool_context()
            with pytest.raises(Exception, match="HTTP Error"):
                tool.run(ctx, "What is the capital of France?")
