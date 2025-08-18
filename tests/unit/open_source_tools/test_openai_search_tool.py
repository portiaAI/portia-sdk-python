"""OpenAI Search tool tests."""

from unittest.mock import Mock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.openai_search_tool import OpenAISearchTool
from tests.utils import get_test_tool_context


def test_openai_search_tool_missing_api_key() -> None:
    """Test that OpenAISearchTool raises ToolHardError if API key is missing."""
    tool = OpenAISearchTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError, match="OPENAI_API_KEY is required"):
            tool.run(ctx, "What is the capital of France?")


def test_openai_search_tool_successful_response() -> None:
    """Test that OpenAISearchTool successfully processes a valid response."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris, a city known for its rich history and culture.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://en.wikipedia.org/wiki/Paris",
                                "title": "Paris - Wikipedia",
                                "start_index": 0,
                                "end_index": 50
                            }
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://britannica.com/place/Paris",
                                "title": "Paris | History, Geography & Culture | Britannica",
                                "start_index": 51,
                                "end_index": 100
                            }
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/france-capital",
                                "title": "France Capital Information",
                                "start_index": 101,
                                "end_index": 150
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }
        ]
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "What is the capital of France?")
            
            # Should return first 3 results (MAX_RESULTS)
            assert len(result) == 3
            assert all("url" in res for res in result)
            assert all("title" in res for res in result)
            assert result[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
            assert result[1]["url"] == "https://britannica.com/place/Paris"
            assert result[2]["url"] == "https://example.com/france-capital"


def test_openai_search_tool_fewer_results_than_max() -> None:
    """Test that OpenAISearchTool successfully processes response with fewer than MAX_RESULTS."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://en.wikipedia.org/wiki/Paris",
                                "title": "Paris - Wikipedia",
                                "start_index": 0,
                                "end_index": 30
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }
        ]
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "What is the capital of France?")
            
            # Should return only 1 result
            assert len(result) == 1
            assert result[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
            assert result[0]["title"] == "Paris - Wikipedia"


def test_openai_search_tool_no_annotations() -> None:
    """Test that OpenAISearchTool handles response with no annotations."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
                    "annotations": []
                },
                "finish_reason": "stop"
            }
        ]
    }

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            result = tool.run(ctx, "What is the capital of France?")
            
            # Should create a basic result with content
            assert len(result) == 1
            assert result[0]["content"] == "The capital of France is Paris."
            assert result[0]["title"] == "Search Results"
            assert result[0]["url"] == ""


def test_openai_search_tool_no_choices() -> None:
    """Test that OpenAISearchTool raises ToolSoftError if no choices in response."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"
    mock_response = {"choices": []}

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = Mock(status_code=200, json=lambda: mock_response)

            with pytest.raises(ToolSoftError, match="No choices in OpenAI response"):
                tool.run(ctx, "What is the capital of France?")


def test_openai_search_tool_http_error() -> None:
    """Test that OpenAISearchTool handles HTTP errors correctly."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_response = Mock(status_code=401, text="Unauthorized")
            mock_post.return_value = mock_response
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized", request=Mock(), response=mock_response
            )

            with pytest.raises(ToolSoftError, match="OpenAI API error: 401"):
                tool.run(ctx, "What is the capital of France?")


def test_openai_search_tool_invalid_json() -> None:
    """Test that OpenAISearchTool handles invalid JSON response."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        ctx = get_test_tool_context()
        with patch("httpx.post") as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            with pytest.raises(ToolSoftError, match="Failed to parse OpenAI response"):
                tool.run(ctx, "What is the capital of France?")


# Async tests for OpenAISearchTool.arun function
@pytest.mark.asyncio
async def test_openai_search_tool_async_missing_api_key() -> None:
    """Test that OpenAISearchTool raises ToolHardError if API key is missing (async)."""
    tool = OpenAISearchTool()
    with patch("os.getenv", return_value=""):
        ctx = get_test_tool_context()
        with pytest.raises(ToolHardError, match="OPENAI_API_KEY is required"):
            await tool.arun(ctx, "What is the capital of France?")


@pytest.mark.asyncio
async def test_openai_search_tool_async_successful_response(httpx_mock: HTTPXMock) -> None:
    """Test that OpenAISearchTool successfully processes a valid response (async)."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://en.wikipedia.org/wiki/Paris",
                                "title": "Paris - Wikipedia",
                                "start_index": 0,
                                "end_index": 30
                            }
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://britannica.com/place/Paris",
                                "title": "Paris | Britannica",
                                "start_index": 31,
                                "end_index": 60
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }
        ]
    }

    with patch("os.getenv", return_value=mock_api_key):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            json=mock_response,
            status_code=200,
        )
        ctx = get_test_tool_context()
        result = await tool.arun(ctx, "What is the capital of France?")
        
        assert len(result) == 2
        assert result[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
        assert result[1]["url"] == "https://britannica.com/place/Paris"


@pytest.mark.asyncio
async def test_openai_search_tool_async_http_error(httpx_mock: HTTPXMock) -> None:
    """Test that OpenAISearchTool handles HTTP errors correctly (async)."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            status_code=500,
            text="Internal Server Error"
        )
        ctx = get_test_tool_context()
        with pytest.raises(ToolSoftError, match="OpenAI API error: 500"):
            await tool.arun(ctx, "What is the capital of France?")


@pytest.mark.asyncio
async def test_openai_search_tool_async_different_query(httpx_mock: HTTPXMock) -> None:
    """Test that OpenAISearchTool works with different search queries (async)."""
    tool = OpenAISearchTool()
    mock_api_key = "sk-test-api-key"
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Joe Biden won the 2020 US Presidential election.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/election-results",
                                "title": "2020 Election Results",
                                "start_index": 0,
                                "end_index": 40
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }
        ]
    }

    with patch("os.getenv", return_value=mock_api_key):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            json=mock_response,
            status_code=200,
        )
        ctx = get_test_tool_context()
        result = await tool.arun(ctx, "Who won the US election in 2020?")
        
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/election-results"
        assert result[0]["title"] == "2020 Election Results"