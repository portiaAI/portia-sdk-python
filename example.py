"""Simple Example."""

from dotenv import load_dotenv

from portia import (
    Config,
    LogLevel,
    Portia,
    example_tool_registry,
)
from portia.cli import CLIExecutionHooks
from portia.config import CONDITIONAL_FEATURE_FLAG

load_dotenv(override=True)
# You can also pass in a clarification handler to manage clarifications
portia = Portia(
    Config.from_default(
        default_log_level=LogLevel.DEBUG,
    ),
    tools=example_tool_registry,
    execution_hooks=CLIExecutionHooks(),
)

plan_run = portia.run(
    "If the temperature in London is above 20C, addd it with the weather in Cairo",
)
