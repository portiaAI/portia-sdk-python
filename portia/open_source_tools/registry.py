"""Example registry containing simple tools."""

from portia.open_source_tools.addition import AdditionTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool_registry import InMemoryToolRegistry

example_tool_registry = InMemoryToolRegistry.from_local_tools(
    [
        AdditionTool(),
        WeatherTool(),
    ],
)
