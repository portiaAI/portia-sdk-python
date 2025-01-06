"""Simple Example."""

from portia.config import default_config
from portia.example_tools.registry import example_tool_registry
from portia.runner import Runner

runner = Runner(
    default_config(),
    tool_registry=example_tool_registry,
)

output = runner.run_query(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)
