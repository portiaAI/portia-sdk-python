"""Simple Example."""

from portia.config import Config, LogLevel
from portia.example_tools.registry import example_tool_registry
from portia.runner import Runner

runner = Runner(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tool_registry=example_tool_registry,
)

output = runner.run_query(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)
