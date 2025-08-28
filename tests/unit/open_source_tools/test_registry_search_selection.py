"""Tests for search tool selection logic in registries."""

import os
from unittest.mock import patch

import pytest

from portia.open_source_tools.openai_search_tool import OpenAISearchTool
from portia.open_source_tools.registry import _get_preferred_search_tool
from portia.open_source_tools.search_tool import SearchTool
from portia.tool_registry import DefaultToolRegistry
from portia.config import Config


class TestSearchToolSelection:
    """Test search tool selection logic."""

    def test_no_api_keys_returns_tavily(self) -> None:
        """Test that Tavily search is returned when no API keys are available."""
        with patch.dict(os.environ, {}, clear=True):
            tool = _get_preferred_search_tool()
            assert isinstance(tool, SearchTool)

    def test_only_openai_key_returns_openai(self) -> None:
        """Test that OpenAI search is returned when only OpenAI key is available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            tool = _get_preferred_search_tool()
            assert isinstance(tool, OpenAISearchTool)

    def test_only_tavily_key_returns_tavily(self) -> None:
        """Test that Tavily search is returned when only Tavily key is available."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}, clear=True):
            tool = _get_preferred_search_tool()
            assert isinstance(tool, SearchTool)

    def test_both_keys_returns_tavily(self) -> None:
        """Test that Tavily search is returned when both keys are available (Tavily takes precedence)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test"}, clear=True):
            tool = _get_preferred_search_tool()
            assert isinstance(tool, SearchTool)


class TestDefaultToolRegistrySearchSelection:
    """Test DefaultToolRegistry search tool selection."""
    
    def _create_test_config(self) -> Config:
        """Create a test config for testing."""
        return Config(
            llm_provider="anthropic",
            default_model="anthropic/claude-3-5-sonnet-20241022",
            anthropic_api_key="sk-ant-test",
        )

    def test_default_registry_no_keys(self) -> None:
        """Test DefaultToolRegistry with no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            config = self._create_test_config()
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should not contain any search tools
            search_tools = [t for t in tools if isinstance(t, (SearchTool, OpenAISearchTool))]
            assert len(search_tools) == 0

    def test_default_registry_only_openai_key(self) -> None:
        """Test DefaultToolRegistry with only OpenAI key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            config = self._create_test_config()
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain OpenAI search tool
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(openai_search_tools) == 1

    def test_default_registry_only_tavily_key(self) -> None:
        """Test DefaultToolRegistry with only Tavily key."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}, clear=True):
            config = self._create_test_config()
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain Tavily search tools
            search_tools = [t for t in tools if isinstance(t, SearchTool)]
            assert len(search_tools) == 1

    def test_default_registry_both_keys(self) -> None:
        """Test DefaultToolRegistry with both keys (Tavily takes precedence)."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "TAVILY_API_KEY": "tvly-test"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = self._create_test_config()
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()

            # Should contain Tavily search tools but not OpenAI
            search_tools = [t for t in tools if isinstance(t, SearchTool)]
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(search_tools) == 1
            assert len(openai_search_tools) == 0