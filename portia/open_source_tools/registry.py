"""Example registry containing simple tools."""

import os
import warnings

from portia.common import validate_extras_dependencies
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.crawl_tool import CrawlTool
from portia.open_source_tools.extract_tool import ExtractTool
from portia.open_source_tools.image_understanding_tool import ImageUnderstandingTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from portia.open_source_tools.map_tool import MapTool
from portia.open_source_tools.openai_search_tool import OpenAISearchTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool_registry import (
    ToolRegistry,
)


def _get_preferred_search_tool():
    """Get the preferred search tool based on available API keys and user preference.
    
    Users can override the default selection by setting PORTIA_SEARCH_PROVIDER to either
    'openai' or 'tavily'. Otherwise, OpenAI search is used if OPENAI_API_KEY is available
    and TAVILY_API_KEY is not.
    """
    from portia.config import Config, SearchProvider
    
    try:
        config = Config()
        has_openai_key = bool(config.openai_api_key and config.openai_api_key.get_secret_value().strip())
        has_tavily_key = bool(os.getenv("TAVILY_API_KEY"))
        search_provider = config.search_provider
    except Exception:
        # Fallback to env vars if config fails
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        has_tavily_key = bool(os.getenv("TAVILY_API_KEY"))
        search_provider_str = os.getenv("PORTIA_SEARCH_PROVIDER", "tavily").lower()
        try:
            search_provider = SearchProvider(search_provider_str)
        except ValueError:
            search_provider = SearchProvider.TAVILY
    
    # If user explicitly sets the provider, honor their choice
    if search_provider == SearchProvider.OPENAI and has_openai_key:
        return OpenAISearchTool()
    elif search_provider == SearchProvider.TAVILY and has_tavily_key:
        return SearchTool()
    
    # Default automatic selection logic
    # If user has OpenAI key but no Tavily key, use OpenAI search
    if has_openai_key and not has_tavily_key:
        return OpenAISearchTool()
    # Otherwise, use Tavily search (default behavior)
    else:
        return SearchTool()

example_tool_registry = ToolRegistry(
    [CalculatorTool(), WeatherTool(), _get_preferred_search_tool(), LLMTool()],
)

open_source_tool_registry = ToolRegistry(
    [
        CalculatorTool(),
        CrawlTool(),
        ExtractTool(),
        FileReaderTool(),
        FileWriterTool(),
        ImageUnderstandingTool(),
        LLMTool(),
        MapTool(),
        _get_preferred_search_tool(),
        WeatherTool(),
    ],
)
if validate_extras_dependencies("tools-browser-local", raise_error=False):
    from .browser_tool import BrowserTool

    open_source_tool_registry.with_tool(BrowserTool())
if validate_extras_dependencies("tools-pdf-reader", raise_error=False) and os.getenv(
    "MISTRAL_API_KEY"
):
    from .pdf_reader_tool import PDFReaderTool

    open_source_tool_registry.with_tool(PDFReaderTool())
