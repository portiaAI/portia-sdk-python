"""Little test script for debugging."""

from portia.runner import Runner, RunnerConfig
from portia.storage import DiskFileStorage
from portia.tool import Tool
from portia.tool_registry import LocalToolRegistry


# Create a local tool
class AdditionTool(Tool):
    """AdditionTool adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b


# Create the ToolRegistry with the tool
tool_registry = LocalToolRegistry.from_local_tools([AdditionTool()])

# Create local storage
storage = DiskFileStorage(storage_dir="/tmp/portia")  # noqa: S108

config = RunnerConfig()
runner = Runner(config=config, tool_registry=tool_registry)
output = runner.run_query("Add 1 and 2")
print(output)  # noqa: T201
