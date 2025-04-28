"""Example registry containing simple tools."""

import logging
import os

from portia.common import validate_extras_dependencies
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.image_understanding_tool import ImageUnderstandingTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.local_file_reader_tool import FileReaderTool
from portia.open_source_tools.local_file_writer_tool import FileWriterTool
from portia.open_source_tools.search_tool import SearchTool
from portia.open_source_tools.weather import WeatherTool
from portia.tool_registry import (
    ToolRegistry,
)

logger = logging.getLogger(__name__)

example_tool_registry = ToolRegistry(
    [CalculatorTool(), WeatherTool(), SearchTool(), LLMTool()],
)

open_source_tool_registry = ToolRegistry(
    [
        CalculatorTool(),
        WeatherTool(),
        SearchTool(),
        LLMTool(),
        FileWriterTool(),
        FileReaderTool(),
        ImageUnderstandingTool(),
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
