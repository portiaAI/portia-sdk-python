"""Unit tests for OpenAI Search Tool."""

import json
import warnings
from unittest.mock import AsyncMock, Mock, patch

import pytest

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.openai_search_tool import OpenAISearchTool
from tests.utils import get_test_tool_context


def test_openai_search_tool_missing_api_key() -> None:
    """Test that OpenAISearchTool raises ToolHardError if API key is missing."""
    with patch("os.getenv", return_value=""):
        with pytest.raises(ToolHardError, match="OPENAI_API_KEY is required"):
            OpenAISearchTool()


def test_openai_search_tool_successful_response() -> None:
    """Test that OpenAISearchTool successfully processes a valid response."""
    mock_api_key = "sk-test-api-key"
    
    # Mock JSON response in the expected format
    json_response = {
        "results": [
            {
                "url": "https://en.wikipedia.org/wiki/Paris",
                "title": "Paris - Wikipedia",
                "content": "The capital of France is Paris, a city known for its rich history and culture."
            },
            {
                "url": "https://britannica.com/place/Paris",
                "title": "Paris | History, Geography & Culture | Britannica",
                "content": "Paris is the capital and most populous city of France."
            },
            {
                "url": "https://example.com/france-capital",
                "title": "France Capital Information",
                "content": "Comprehensive information about France's capital city."
            }
        ]
    }
    
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = json.dumps(json_response)

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = tool.run(ctx, "What is the capital of France?")
            
            # Should return all results (no MAX_RESULTS limit anymore)
            assert len(result) == 3
            assert all("url" in res for res in result)
            assert all("title" in res for res in result)
            assert result[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
            assert result[1]["url"] == "https://britannica.com/place/Paris"
            assert result[2]["url"] == "https://example.com/france-capital"


def test_openai_search_tool_fewer_results_than_max() -> None:
    """Test that OpenAISearchTool successfully processes response with fewer results."""
    mock_api_key = "sk-test-api-key"
    
    # Mock JSON response with single result
    json_response = {
        "results": [
            {
                "url": "https://en.wikipedia.org/wiki/Paris",
                "title": "Paris - Wikipedia",
                "content": "The capital of France is Paris."
            }
        ]
    }
    
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = json.dumps(json_response)

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = tool.run(ctx, "What is the capital of France?")
            
            # Should return only 1 result
            assert len(result) == 1
            assert result[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
            assert result[0]["title"] == "Paris - Wikipedia"


def test_openai_search_tool_no_annotations() -> None:
    """Test that OpenAISearchTool handles response with empty results."""
    mock_api_key = "sk-test-api-key"
    
    # Mock JSON response with empty results
    json_response = {"results": []}
    
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = json.dumps(json_response)

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="No search results found in OpenAI response"):
                tool.run(ctx, "What is the capital of France?")


# Async tests for OpenAISearchTool.arun function
@pytest.mark.asyncio
async def test_openai_search_tool_async_missing_api_key() -> None:
    """Test that OpenAISearchTool raises ToolHardError if API key is missing (async)."""
    with patch("os.getenv", return_value=""):
        with pytest.raises(ToolHardError, match="OPENAI_API_KEY is required"):
            OpenAISearchTool()


@pytest.mark.asyncio
async def test_openai_search_tool_async_api_error() -> None:
    """Test that OpenAISearchTool handles API errors correctly (async)."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.AsyncOpenAI") as mock_async_openai:
            mock_client = Mock()
            
            # Create an async mock that raises an exception
            async def mock_create(*args, **kwargs):
                raise Exception("API error occurred")
            mock_client.responses.create = mock_create
            mock_async_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="OpenAI API error"):
                await tool.arun(ctx, "What is the capital of France?")


@pytest.mark.asyncio
async def test_openai_search_tool_async_successful_response() -> None:
    """Test that OpenAISearchTool successfully processes a valid response (async)."""
    mock_api_key = "sk-test-api-key"
    
    # Mock JSON response in the expected format
    json_response = {
        "results": [
            {
                "url": "https://en.wikipedia.org/wiki/Paris",
                "title": "Paris - Wikipedia",
                "content": "The capital of France is Paris."
            },
            {
                "url": "https://britannica.com/place/Paris",
                "title": "Paris | Britannica",
                "content": "Paris is the capital city of France."
            }
        ]
    }
    
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = json.dumps(json_response)

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.AsyncOpenAI") as mock_async_openai:
            mock_client = Mock()
            # Create an async mock
            async def mock_create(*args, **kwargs):
                return mock_response
            mock_client.responses.create = mock_create
            mock_async_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = await tool.arun(ctx, "What is the capital of France?")

            assert len(result) == 2
            assert result[0]["url"] == "https://en.wikipedia.org/wiki/Paris"
            assert result[1]["url"] == "https://britannica.com/place/Paris"


@pytest.mark.asyncio
async def test_openai_search_tool_async_different_query() -> None:
    """Test that OpenAISearchTool works with different search queries (async)."""
    mock_api_key = "sk-test-api-key"
    
    # Mock JSON response for different query
    json_response = {
        "results": [
            {
                "url": "https://example.com/election-results",
                "title": "2020 Election Results",
                "content": "Joe Biden won the 2020 US Presidential election."
            }
        ]
    }
    
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = json.dumps(json_response)

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.AsyncOpenAI") as mock_async_openai:
            mock_client = Mock()
            # Create an async mock
            async def mock_create(*args, **kwargs):
                return mock_response
            mock_client.responses.create = mock_create
            mock_async_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = await tool.arun(ctx, "Who won the US election in 2020?")

            assert len(result) == 1
            assert result[0]["url"] == "https://example.com/election-results"
            assert result[0]["title"] == "2020 Election Results"


def test_openai_search_tool_no_search_results() -> None:
    """Test that OpenAISearchTool handles invalid JSON gracefully."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response with invalid JSON that will fall back to basic response
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = "Invalid JSON response"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = tool.run(ctx, "What is the capital of France?")
            
            # Should fall back to basic response
            assert len(result) == 1
            assert result[0]["title"] == "Search Results"
            assert result[0]["content"] == "Invalid JSON response"


def test_openai_search_tool_different_query() -> None:
    """Test that OpenAISearchTool works with different search queries."""
    mock_api_key = "sk-test-api-key"
    
    # Mock JSON response for different query
    json_response = {
        "results": [
            {
                "url": "https://example.com/election-results",
                "title": "2020 Election Results",
                "content": "Joe Biden won the 2020 US Presidential election."
            }
        ]
    }
    
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.text = json.dumps(json_response)

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = tool.run(ctx, "Who won the US election in 2020?")

            assert len(result) == 1
            assert result[0]["url"] == "https://example.com/election-results"
            assert result[0]["title"] == "2020 Election Results"


def test_openai_search_tool_api_error() -> None:
    """Test that OpenAISearchTool handles API errors correctly."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.side_effect = Exception("API error occurred")
            mock_openai.return_value = mock_client

            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="OpenAI API error"):
                tool.run(ctx, "What is the capital of France?")