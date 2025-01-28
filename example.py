"""Simple Example."""

from portia.config import Config, LogLevel
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry

runner = Runner(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tool_registry=PortiaToolRegistry(Config.from_default(default_log_level=LogLevel.DEBUG)),
)


# Simple Example
workflow = runner.execute_query(
    "get the weather in Cairo and Alex and find me the best activity to do in both",
)

