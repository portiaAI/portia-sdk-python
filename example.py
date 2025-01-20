"""Simple Example."""

from portia.config import Config, LogLevel, StorageClass
from portia.open_source_tools.registry import example_tool_registry
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry

my_config = Config.from_default(
    default_log_level=LogLevel.DEBUG,
    storage_class=StorageClass.CLOUD,
)

tool_registry = example_tool_registry + PortiaToolRegistry(my_config)

runner = Runner(
    my_config,
    tool_registry=tool_registry,
)


# Simple Example
workflow = runner.execute_query(
    "Get the weather in Cairo and Alexandria, then add them together",
)
