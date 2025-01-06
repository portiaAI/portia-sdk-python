"""Example registry containing simple tools."""

from portia.example_tools.addition import AdditionTool
from portia.example_tools.weather import WeatherTool
from portia.tool_registry import InMemoryToolRegistry

example_tool_registry = InMemoryToolRegistry.from_local_tools(
    [
        AdditionTool(),
        WeatherTool(),
    ],
)
