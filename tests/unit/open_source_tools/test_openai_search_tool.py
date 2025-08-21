"""OpenAI Search tool tests."""

import json
import os
from unittest.mock import Mock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.openai_search_tool import OpenAISearchTool
from tests.utils import get_test_tool_context


# Original tests for backward compatibility and integration with Response API
def test_openai_search_tool_missing_api_key() -> None:
    """Test that OpenAISearchTool raises ToolHardError if API key is missing."""
    with patch("os.getenv", return_value=""):
        with pytest.raises(ToolHardError, match="OPENAI_API_KEY is required"):
            OpenAISearchTool()


def test_openai_search_tool_successful_response() -> None:
    """Test that OpenAISearchTool successfully processes a valid response."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response object structure
    mock_output_item = Mock()
    mock_output_item.annotations = [
        Mock(
            type="url_citation",
            url_citation=Mock(
                url="https://en.wikipedia.org/wiki/Paris",
                title="Paris - Wikipedia"
            )
        ),
        Mock(
            type="url_citation", 
            url_citation=Mock(
                url="https://britannica.com/place/Paris",
                title="Paris | History, Geography & Culture | Britannica"
            )
        ),
        Mock(
            type="url_citation",
            url_citation=Mock(
                url="https://example.com/france-capital",
                title="France Capital Information"
            )
        )
    ]
    mock_output_item.text = "The capital of France is Paris, a city known for its rich history and culture."
    
    mock_response = Mock()
    mock_response.output = [mock_output_item]
    mock_response.output_text = "The capital of France is Paris."

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
    
    # Mock response object structure
    mock_output_item = Mock()
    mock_output_item.annotations = [
        Mock(
            type="url_citation",
            url_citation=Mock(
                url="https://en.wikipedia.org/wiki/Paris",
                title="Paris - Wikipedia"
            )
        )
    ]
    mock_output_item.text = "The capital of France is Paris."
    
    mock_response = Mock()
    mock_response.output = [mock_output_item]
    mock_response.output_text = "The capital of France is Paris."

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
    """Test that OpenAISearchTool handles response with no annotations."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response with no annotations
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.annotations = []
    mock_response.output_text = "The capital of France is Paris."

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            result = tool.run(ctx, "What is the capital of France?")
            
            # Should create a basic result with content
            assert len(result) == 1
            assert result[0]["content"] == "The capital of France is Paris."
            assert result[0]["title"] == "Search Results"
            assert result[0]["url"] == ""


def test_openai_search_tool_no_output() -> None:
    """Test that OpenAISearchTool raises ToolSoftError if no output in response."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response with no output
    mock_response = Mock()
    mock_response.output = None

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="No output in OpenAI response"):
                tool.run(ctx, "What is the capital of France?")


def test_openai_search_tool_http_error() -> None:
    """Test that OpenAISearchTool handles HTTP errors correctly."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            # Simulate 401 authentication error
            from openai import AuthenticationError
            mock_client.responses.create.side_effect = AuthenticationError(
                message="Incorrect API key provided",
                response=Mock(status_code=401),
                body=None
            )
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolHardError, match="Invalid OpenAI API key"):
                tool.run(ctx, "What is the capital of France?")


def test_openai_search_tool_api_error() -> None:
    """Test that OpenAISearchTool handles general API errors."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            # Simulate general API error
            mock_client.responses.create.side_effect = Exception("General API error")
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="OpenAI API error"):
                tool.run(ctx, "What is the capital of France?")


# Async tests for OpenAISearchTool.arun function
@pytest.mark.asyncio
async def test_openai_search_tool_async_missing_api_key() -> None:
    """Test that OpenAISearchTool raises ToolHardError if API key is missing (async)."""
    with patch("os.getenv", return_value=""):
        with pytest.raises(ToolHardError, match="OPENAI_API_KEY is required"):
            OpenAISearchTool()


@pytest.mark.asyncio
async def test_openai_search_tool_async_successful_response() -> None:
    """Test that OpenAISearchTool successfully processes a valid response (async)."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response object structure
    mock_output_item = Mock()
    mock_output_item.annotations = [
        Mock(
            type="url_citation",
            url_citation=Mock(
                url="https://en.wikipedia.org/wiki/Paris",
                title="Paris - Wikipedia"
            )
        ),
        Mock(
            type="url_citation", 
            url_citation=Mock(
                url="https://britannica.com/place/Paris",
                title="Paris | Britannica"
            )
        )
    ]
    mock_output_item.text = "The capital of France is Paris."
    
    mock_response = Mock()
    mock_response.output = [mock_output_item]
    mock_response.output_text = "The capital of France is Paris."

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
async def test_openai_search_tool_async_http_error() -> None:
    """Test that OpenAISearchTool handles HTTP errors correctly (async)."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.AsyncOpenAI") as mock_async_openai:
            mock_client = Mock()
            # Create an async mock that raises server error
            async def mock_create_error(*args, **kwargs):
                raise Exception("500 Internal Server Error")
            mock_client.responses.create = mock_create_error
            mock_async_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="OpenAI API error"):
                await tool.arun(ctx, "What is the capital of France?")


def test_openai_search_tool_manual_override_tavily_with_tavily_key() -> None:
    """Test manual override to Tavily when Tavily key is available."""
    tool = OpenAISearchTool()
    from portia.open_source_tools.registry import _get_preferred_search_tool
    from portia.open_source_tools.search_tool import SearchTool
    
    env_vars = {
        "OPENAI_API_KEY": "sk-test",
        "TAVILY_API_KEY": "tvly-test", 
        "PORTIA_SEARCH_PROVIDER": "tavily"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        selected_tool = _get_preferred_search_tool()
        assert isinstance(selected_tool, SearchTool)


def test_registry_return_search_tool_line_coverage() -> None:
    """Test that covers the specific return SearchTool() line in registry.py:37."""
    from portia.open_source_tools.registry import _get_preferred_search_tool
    from portia.open_source_tools.search_tool import SearchTool
    
    # Set up environment to specifically trigger line 37: return SearchTool()
    env_vars = {
        "TAVILY_API_KEY": "tvly-test", 
        "PORTIA_SEARCH_PROVIDER": "tavily"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        selected_tool = _get_preferred_search_tool()
        # This should trigger the return SearchTool() on line 37
        assert isinstance(selected_tool, SearchTool)


@pytest.mark.asyncio
async def test_openai_search_tool_async_different_query() -> None:
    """Test that OpenAISearchTool works with different search queries (async)."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response object structure
    mock_output_item = Mock()
    mock_output_item.annotations = [
        Mock(
            type="url_citation",
            url_citation=Mock(
                url="https://example.com/election-results",
                title="2020 Election Results"
            )
        )
    ]
    mock_output_item.text = "Joe Biden won the 2020 US Presidential election."
    
    mock_response = Mock()
    mock_response.output = [mock_output_item]
    mock_response.output_text = "Joe Biden won the 2020 US Presidential election."

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
    """Test that OpenAISearchTool raises ToolSoftError when no search results found."""
    mock_api_key = "sk-test-api-key"
    
    # Mock response with empty content and no annotations
    mock_response = Mock()
    mock_response.output = Mock()
    mock_response.output.annotations = []
    mock_response.output_text = ""

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="No search results found in OpenAI response"):
                tool.run(ctx, "What is the capital of France?")

def test_openai_search_tool_rate_limit_error() -> None:
    """Test that OpenAISearchTool handles rate limit errors correctly."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            # Simulate rate limit error
            mock_client.responses.create.side_effect = Exception("429 Rate limit exceeded")
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="OpenAI API rate limit exceeded"):
                tool.run(ctx, "What is the capital of France?")


def test_openai_search_tool_server_error() -> None:
    """Test that OpenAISearchTool handles server errors correctly."""
    mock_api_key = "sk-test-api-key"

    with patch("os.getenv", return_value=mock_api_key):
        with patch("portia.open_source_tools.openai_search_tool.OpenAI") as mock_openai:
            mock_client = Mock()
            # Simulate server error
            mock_client.responses.create.side_effect = Exception("500 Internal Server Error")
            mock_openai.return_value = mock_client
            
            tool = OpenAISearchTool()
            ctx = get_test_tool_context()
            with pytest.raises(ToolSoftError, match="OpenAI API server error"):
                tool.run(ctx, "What is the capital of France?")