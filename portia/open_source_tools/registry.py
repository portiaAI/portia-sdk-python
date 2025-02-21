"""Example registry containing simple tools."""

import logging
import os
import re

from portia.config import Config
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool import Tool
from portia.tool_registry import (
    InMemoryToolRegistry,
    PortiaToolRegistry,
    ToolRegistry,
)

logger = logging.getLogger(__name__)

example_tool_registry = InMemoryToolRegistry.from_local_tools(
    [CalculatorTool(), WeatherTool(), SearchTool()],
)


open_source_tool_registry = InMemoryToolRegistry.from_local_tools(
    [
        CalculatorTool(),
        WeatherTool(),
        SearchTool(),
        LLMTool(),
        FileWriterTool(),
        FileReaderTool(),
    ],
)


EXCLUDED_BY_DEFAULT_TOOL_REGEXS: frozenset[str] = frozenset(
    {
        # Exclude Outlook by default as it clashes with Gmail
        "portia:microsoft:outlook:*",
    },
)


def get_default_tool_registry(config: Config) -> ToolRegistry:
    """Get the default tool registry based on the configuration.

    This includes the following tools:
    - All open source tools that don't require API keys
    - Search tool if you have a Tavily API key
    - Weather tool if you have an OpenWeatherMap API key
    - Portia cloud tools if you have a Portia cloud API key
    """

    def default_tool_filter(tool: Tool) -> bool:
        """Filter to get the default set of tools offered by Portia cloud."""
        return not any(re.match(regex, tool.id) for regex in EXCLUDED_BY_DEFAULT_TOOL_REGEXS)

    tool_registry = InMemoryToolRegistry.from_local_tools(
        [
            CalculatorTool(),
            LLMTool(),
            FileWriterTool(),
            FileReaderTool(),
        ],
    )
    if os.getenv("TAVILY_API_KEY"):
        tool_registry.register_tool(SearchTool())
    if not os.getenv("OPENWEATHERMAP_API_KEY"):
        tool_registry.register_tool(WeatherTool())
    if config.portia_api_key:
        tool_registry += PortiaToolRegistry(config).filter_tools(default_tool_filter)

    return tool_registry
