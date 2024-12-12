# Portia SDK Python


## Usage

### Installation

```bash
pip install portia-sdk-python 
```

### Simple Usage

```python
# Create a local tool
class AddTool(Tool):
    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"

    def run(self, a: int, b: int) -> int:
        return a + b


# run a query:
runner = Runner([AddTool])
runner.run_query("Add 1 and 2")
```

### With Custom Registries

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
tool_registry = LocalToolRegistry.from_local_tools([AdditionTool()])

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

## Tests

Run tests with `poetry run pytest`.

## Release

Releases are controlled via Github Actions and the version field of the `pyproject.toml`. To release:

1. Create a PR that updates the version field in the `pyproject.toml`.
2. Merge the PR to main.
3. Github Actions will create a new tag and push the new version to PyPi.