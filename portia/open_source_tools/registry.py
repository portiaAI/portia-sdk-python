"""Example registry containing simple tools."""

from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool_registry import InMemoryToolRegistry

example_tool_registry = InMemoryToolRegistry.from_local_tools(
    [
        CalculatorTool(),
        WeatherTool(),
        SearchTool(),
    ],
)
