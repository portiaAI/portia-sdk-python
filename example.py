"""Simple Example."""

from __future__ import annotations

from portia.config import default_config
from portia.example_tools.registry import example_tool_registry
from portia.runner import Runner

config = default_config()
config.default_log_level = "debug"

runner = Runner(
    config,
    tool_registry=example_tool_registry,
)

output = runner.run_query(
    "Get the temperature in London and Sydney and then add the two temperatures together "
    "rounded to two decimal places.",
)
