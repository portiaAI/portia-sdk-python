from portia.plan_storage import DiskFileStorage
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import LocalToolRegistry


# Create a local tool
class AddTool(Tool):
    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        return a + b


# Create the ToolRegistry with the tool
tool_registry = LocalToolRegistry.from_local_tools([AddTool()])

# Create local storage
storage = DiskFileStorage(storage_dir="/tmp/portia")

runner = Runner(tool_registry=tool_registry)
output = runner.run_query("Add 1 and 2")
print(output)
