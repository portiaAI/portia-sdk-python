"""Example registry containing simple tools."""

from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool_registry import InMemoryToolRegistry

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
