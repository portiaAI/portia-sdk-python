# Portia SDK Python


## Usage

### Installation

```bash
pip install portia-sdk-python 
```

### Simple Usage

```python
# Create a local tool
<<<<<<< HEAD
class AdditionTool(Tool):
=======
class AddTool(Tool):
>>>>>>> 89ab650 (wip)
    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        return a + b


# run a query:
<<<<<<< HEAD
runner = Runner([AdditionTool])
=======
runner = Runner([AddTool])
>>>>>>> 89ab650 (wip)
runner.run_query("Add 1 and 2")
```

### With Custom Registries

```python
from portia.tool import Tool
from portia.tool_registry import LocalToolRegistry

# Create a local tool
class AdditionTool(Tool):
    id: str = "addition_tool"
    name: str = "Addition Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        return a + b


# Create the ToolRegistry with the tool
<<<<<<< HEAD
tool_registry = LocalToolRegistry.from_local_tools([AdditionTool()])
=======
tool_registry = LocalToolRegistry.from_local_tools([AddTool()])
>>>>>>> 89ab650 (wip)

# Create local storage
storage = LocalStorage()

runner = Runner(storage=storage, tool_registry=tool_registry)
runner.run_query("Add 1 and 2")
```

### Using Portia Cloud

```python
runner = Runner(api_key='123')
runner.run_query("Add 1 and 2")
```

### Hybrid Approach

Multiple registries can be combined to give the power of Portia Cloud with the customization of local tools:

```python
from portia.tool import Tool
from portia.tool_registry import LocalToolRegistry

# Create a local tool
class AdditionTool(Tool):
    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        return a + b


# Create the ToolRegistry with the tool
local_tool_registry = LocalToolRegistry.from_local_tools([AdditionTool()])

remote_tool_registry = PortiaToolRegistry(api_key="123")

tool_registry = local_tool_registry + remote_tool_registry

# Create local storage
storage = LocalStorage()

runner = Runner(storage=storage, tool_registry=tool_registry)
runner.run_query("Add 1 and 2")
```


## Tests

Run tests with `poetry run pytest`.

## Release

Releases are controlled via Github Actions and the version field of the `pyproject.toml`. To release:

1. Create a PR that updates the version field in the `pyproject.toml`.
2. Merge the PR to main.
3. Github Actions will create a new tag and push the new version to PyPi.