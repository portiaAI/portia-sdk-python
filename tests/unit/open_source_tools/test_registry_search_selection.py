"""Tests for search tool selection logic in registries."""

import os
import warnings
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

    def test_manual_override_openai(self) -> None:
        """Test manual override to use OpenAI search."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "TAVILY_API_KEY": "tvly-test",
            "PORTIA_SEARCH_PROVIDER": "openai"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            tool = _get_preferred_search_tool()
            assert isinstance(tool, OpenAISearchTool)

    def test_manual_override_tavily(self) -> None:
        """Test manual override to use Tavily search."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "PORTIA_SEARCH_PROVIDER": "tavily"
        }
        # Note: Tavily key is missing, but user explicitly wants Tavily
        # This should fall back to default logic
        with patch.dict(os.environ, env_vars, clear=True):
            tool = _get_preferred_search_tool()
            # Should fall back to OpenAI since Tavily key is missing
            assert isinstance(tool, OpenAISearchTool)

    def test_manual_override_openai_without_key(self) -> None:
        """Test manual override to OpenAI without OpenAI key falls back to default."""
        env_vars = {
            "TAVILY_API_KEY": "tvly-test",
            "PORTIA_SEARCH_PROVIDER": "openai"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            tool = _get_preferred_search_tool()
            # Should fall back to Tavily since OpenAI key is missing
            assert isinstance(tool, SearchTool)

    def test_invalid_provider_warning(self) -> None:
        """Test that invalid provider value shows warning and falls back to default."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "PORTIA_SEARCH_PROVIDER": "invalid"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tool = _get_preferred_search_tool()
                
                # Should fall back to OpenAI (default logic)
                assert isinstance(tool, OpenAISearchTool)
                
                # Should have issued a warning
                assert len(w) == 1
                assert "Invalid search_provider" in str(w[0].message)
                assert "invalid" in str(w[0].message)


class TestDefaultToolRegistrySearchSelection:
    """Test search tool selection in DefaultToolRegistry."""

    def test_default_registry_no_keys(self) -> None:
        """Test DefaultToolRegistry with no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(portia_api_key=None, llm_provider="openai")
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should not contain any search tools
            search_tools = [t for t in tools if "search" in t.id.lower()]
            assert len(search_tools) == 0

    def test_default_registry_only_openai_key(self) -> None:
        """Test DefaultToolRegistry with only OpenAI key."""
        config = Config(portia_api_key=None, llm_provider="openai")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain OpenAI search tool
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(openai_search_tools) == 1
            
            # Should not contain Tavily tools
            tavily_search_tools = [t for t in tools if isinstance(t, SearchTool)]
            assert len(tavily_search_tools) == 0

    def test_default_registry_only_tavily_key(self) -> None:
        """Test DefaultToolRegistry with only Tavily key."""
        config = Config(portia_api_key=None, llm_provider="openai")
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}, clear=True):
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain Tavily search tool and related tools
            tavily_search_tools = [t for t in tools if isinstance(t, SearchTool)]
            assert len(tavily_search_tools) == 1
            
            # Should also have map, extract, crawl tools
            tool_ids = [t.id for t in tools]
            assert "search_tool" in tool_ids
            assert "map_tool" in tool_ids
            assert "extract_tool" in tool_ids
            assert "crawl_tool" in tool_ids
            
            # Should not contain OpenAI search tool
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(openai_search_tools) == 0

    def test_default_registry_both_keys_prefers_tavily(self) -> None:
        """Test DefaultToolRegistry with both keys prefers Tavily."""
        config = Config(portia_api_key=None, llm_provider="openai")
        env_vars = {"OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test"}
        with patch.dict(os.environ, env_vars, clear=True):
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain Tavily tools
            tavily_search_tools = [t for t in tools if isinstance(t, SearchTool)]
            assert len(tavily_search_tools) == 1
            
            # Should not contain OpenAI search tool
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(openai_search_tools) == 0

    def test_default_registry_manual_override_openai(self) -> None:
        """Test DefaultToolRegistry with manual override to OpenAI."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "TAVILY_API_KEY": "tvly-test",
            "PORTIA_SEARCH_PROVIDER": "openai"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config(portia_api_key=None, llm_provider="openai")
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain OpenAI search tool
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(openai_search_tools) == 1
            
            # Should not contain Tavily tools
            tavily_search_tools = [t for t in tools if isinstance(t, SearchTool)]
            assert len(tavily_search_tools) == 0
            
            # Should not have map, extract, crawl tools
            tool_ids = [t.id for t in tools]
            assert "map_tool" not in tool_ids
            assert "extract_tool" not in tool_ids
            assert "crawl_tool" not in tool_ids

    def test_default_registry_manual_override_tavily(self) -> None:
        """Test DefaultToolRegistry with manual override to Tavily."""
        config = Config(portia_api_key=None, llm_provider="openai")
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "TAVILY_API_KEY": "tvly-test",
            "PORTIA_SEARCH_PROVIDER": "tavily"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            registry = DefaultToolRegistry(config)
            tools = registry.get_tools()
            
            # Should contain Tavily tools
            tavily_search_tools = [t for t in tools if isinstance(t, SearchTool)]
            assert len(tavily_search_tools) == 1
            
            # Should not contain OpenAI search tool
            openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
            assert len(openai_search_tools) == 0

    def test_default_registry_invalid_provider_warning(self) -> None:
        """Test DefaultToolRegistry with invalid provider shows warning."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "PORTIA_SEARCH_PROVIDER": "invalid"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = Config(portia_api_key=None, llm_provider="openai")
                registry = DefaultToolRegistry(config)
                tools = registry.get_tools()
                
                # Should fall back to OpenAI (default logic)
                openai_search_tools = [t for t in tools if isinstance(t, OpenAISearchTool)]
                assert len(openai_search_tools) == 1
                
                # Should have issued a warning
                assert len(w) == 1
                assert "Invalid search_provider" in str(w[0].message)
                assert "invalid" in str(w[0].message)