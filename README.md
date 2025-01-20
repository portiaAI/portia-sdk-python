# Portia SDK Python

[Mounir notes] Suggested structure:
- Overview:
    - What it is in a sentence or two
    - What's in this repo
    - Contributor guide link
- Why Portia
    - Plan in an explicit, structured way
    - Manage workflow execution statefully
    - Authenticate tool calls just-in-time
    - Direct execution with Clarifications
- What you can build on Portia
- Let's get started!
    - Install -- validate installation with a simple example using `run_query` and example tools
    - Demonstrate the key features in one multi-tool, multi-step example -- Simple example with `generate_plan`, `create_workflow`, `execute_workflow` (e.g. something with weather and file writing)
- Where can I learn more
    - Docs
    - Other 

## Usage

### Installation

```bash
pip install portia-sdk-python 
```


### Simple Usage with Example Registry

```python
from portia.config import default_config
from portia.open_source_tools.registry import example_tool_registry
from portia.runner import Runner


runner = Runner(
    config=default_config(),
    tool_registry=example_tool_registry,
)

workflow = runner.execute_query("Add 1 and 2")
```


### With Custom Local Tools

```python
from portia.config import default_config
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import InMemoryToolRegistry

# Create a local tool
class SubtractionTool(Tool):
    id: str = "subtraction_tool"
    name: str = "Subtraction Tool"
    description: str = "Takes two numbers and subtracts them together"

    def run(self, a: int, b: int) -> int:
        return a - b


# Create the ToolRegistry with the tool
tool_registry = InMemoryToolRegistry.from_local_tools([SubtractionTool()])

runner = Runner(config=default_config(), tool_registry=tool_registry)
workflow = runner.execute_query("Subtract 1 and 2")
```

### Hybrid Approach

Multiple registries can be combined to give the power of Portia Cloud with the customization of local tools:

```python

from portia.config import default_config
from portia.open_source_tools.registry import example_tool_registry
from portia.runner import Runner

config = default_config()

remote_registry = PortiaToolRegistry(
    config=config,
)

registry = example_tool_registry + remote_registry

runner = Runner(
    config,
    tool_registry=registry,
)

runner.execute_query("Get the weather in Sydney and London then email me with a summary at hello@portialabs.ai")
```


## Tests

We write two types of tests:
- Unit tests should mock out the LLM providers, and aim to give quick feedback. They should mock out LLM providers.
- Integration tests actually call LLM providers, are much slower but test the system works fully.

To run tests:
- Run all tests with `poetry run pytest`.
- Run unit tests with `poetry run pytest tests/unit`.
- Run integration tests with `poetry run pytest tests/integration`.

We utilize [pytest-parallel](https://pypi.org/project/pytest-parallel/) to execute tests in parallel. You can add the `--workers=4` argument to the commands above to run in parallel. If you run into issues running this try setting `export NO_PROXY=true` first.

## Release

Releases are controlled via Github Actions and the version field of the `pyproject.toml`. To release:

1. Create a PR that updates the version field in the `pyproject.toml`.
2. Merge the PR to main.
3. Github Actions will create a new tag and push the new version to PyPi.

## CLI 

To test the CLI locally run 

```bash
pip install -e . 
export OPENAI_API_KEY=$KEY
portia-cli run "add 4 + 8"
```

## Logging

Custom tools can make use of the portia logging:

```python
from portia.logging import logger

class AdditionTool(Tool):
    """Adds two numbers."""
    def run(self, a: float, b: float) -> float | InputClarification:
        """Add the numbers."""
        logger.debug(f"Adding {a} and {b}")
        return a + b

```

The logging implementation itself can also be overridden by any logger that fulfils the LoggerInterface.

For example to use the built in python logger:

```python
import logging
from portia.logging import logger_manager

logger = logging.getLogger(__name__)

logger_manager.set_logger(logger)
```
