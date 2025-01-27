# Portia SDK Python

Welcome to our repository! Portia AI is an open source developer framework for stateful, authenticated agentic workflows. The core product accessible in this repository is extensible with our complimentary cloud features which are aimed at making production deployments easier and faster.
Play around, break things and tell us how you're getting on in our <a href="https://discord.gg/DvAJz9ffaR" target="_blank">**Discord channel (↗)**</a>. Most importantly please be kind to your fellow humans (<a href="https://github.com/portiaAI/portia-sdk-python/blob/main/CODE_OF_CONDUCT.md" target="_blank" rel="noopener noreferrer">**Code of Conduct (↗)**</a>). 


## Why Portia AI
| Problem | Portia's answer |
| ------- | --------------- |
| **Planning:** Many use cases require visibility into the LLM’s reasoning, particularly for complex tasks requiring multiple steps and tools. LLMs also struggle picking the right tools as their tool set grows: a recurring limitation for production deployments | **Multi-agent plans:** Our open source, multi-shot prompter guides your LLM to produce a [`Plan`](https://docs.portia.dev/generate-plan) in response to a prompt, weaving the relevant tools, inputs and outputs for every step. |
| **Execution:** Tracking an LLM’s progress mid-task is difficult, making it harder to intervene when guidance is needed. This is especially critical for enforcing company policies or correcting hallucinations (hello, missing arguments in tool calls!) | **Stateful workflows:** Portia will spin up a multi-agent [`Workflow`](https://docs.portia.dev/execute-workflow) to execute on generated plans and track their state throughout execution. Using our [`Clarification`](https://docs.portia.dev/manage-clarifications) abstraction you can define points where you want to take control of workflow execution e.g. to resolve missing information or multiple choice decisions. Portia serialises the workflow state, and you can manage its storage / retrieval yourself or use our cloud offering for simplicity. |
| **Authentication:** Existing solutions often disrupt the user experience with cumbersome authentication flows or require pre-emptive, full access to every tool—an approach that doesn’t scale for multi-agent assistants. | **Extensible, authenticated tool calling:** Bring your own tools on our extensible [`Tool`](https://docs.portia.dev/extend-tool-definitions) abstraction, or use our growing plug and play authenticated [tool library](https://docs.portia.dev/run-portia-tools), which will include a number of popular SaaS providers over time (Google, Zendesk, Hubspot, Github etc.). All Portia tools feature just-in-time authentication with token refresh, offering security without compromising on user experience. |


## Quickstart

### Installation

1. Install the Portia Python SDK
```bash
pip install portia-sdk-python 
```
2. Ensure you have an API key set up
```bash
export OPENAI_API_KEY='your-api-key-here'
```
3. Validate your installation by submitting a simple maths prompt from the command line
```
portia-cli run "add 1 + 2"
```
>[!NOTE]
> We support Anthropic and Mistral AI as well and we're working on adding more models asap. For now if you want to use either model you'd have to set up the relevant API key and add one of these args to your CLI command:<br/>
> `portia-cli run --llm-provider="anthropic" "add 1 + 2"` or `portia-cli run --llm-provider="mistralai" "add 1 + 2"`

**All set? Now let's explore some basic usage of the product 🚀**


### E2E example with open source tools
This example is meant to get you familiar with a few of our core abstractions:
- A `Plan` is the set of steps an LLM thinks it should take in order to respond to a user prompt. They immutable, structured and human-readable.
- A `Workflow` is a unique instantiation of a `Plan`. The purpose of a `Workflow` is to capture the state of a unique plan run at every step in an auditable way.
- A `Runner` is the main orchestrator of plan generation. It is also capable of workflow creation, execution, pausing and resumption.

Before running the code below, make sure you have the following keys set as environment variables in your .env file:
- An OpenAI API key (or other LLM API key) set as `OPENAI_API_KEY=`
- A Tavily <a href="https://tavily.com/" target="_blank">(**↗**)</a> API key set as `TAVILY_API_KEY=`

```python
from dotenv import load_dotenv
from portia.runner import Runner
from portia.config import default_config
from portia.open_source_tools.registry import example_tool_registry

load_dotenv()

# Instantiate a Portia runner. Load it with the default config and with the example tools.
runner = Runner(config=default_config(), tool_registry=example_tool_registry)

# Generate the plan from the user query
plan = runner.generate_plan('Which stock price grew faster in 2024, Amazon or Google?')
print(plan.model_dump_json(indent=2))

# Create and execute the workflow from the generated plan
workflow = runner.create_workflow(plan)
workflow = runner.execute_workflow(workflow)

# Serialise into JSON and print the output
print(workflow.model_dump_json(indent=2))
```

### E2E example with Portia cloud storage
Our cloud offering will allow you to easily store and retrieve workflows in the Portia cloud, access our library of cloud hosted tools, and use the Portia dashboard to view workflow, clarification and tool call logs. Head over to <a href="apps.portialabs.ai" target="_blank">**apps.portialabs.ai (↗)**</a> and get your Portia API key. You will need to set it as the env variable `PORTIA_API_KEY`.<br/>
Note that this example also requires the environment variables `OPENAI_API_KEY` (or ANTHROPIC or MISTRALAI if you're using either) and `TAVILY_API_KEY` as the [previous one](#e2e-example-with-open-source-tools).

The example below introduces **some** of the config options available with Portia AI:
- The `storage_class` is set using the `StorageClass` ENUM to 'CLOUD`. So long as your `PORTIA_API_KEY` is set, workflows and tool calls will be logged and appear automatically in your Portia dashboard at <a href="apps.portialabs.ai" target="_blank">**apps.portialabs.ai (↗)**</a>.
- The `default_log_level` is set using the `LogLevel` ENUM to `DEBUG` so you can get some insight into the sausage factory in your terminal, including plan generation, workflow states, tool calls and outputs at every step 😅
- The `llm_provider`, `llm_model` and `xxx_api_key` (varies depending on model provider chosen) are used to choose the specific LLM provider and model. In the example below we're splurging and using GPT 4.0!

Finally we also introduce the concept of a `tool_registry`, which is a flexible grouping of tools.

```python
import os
from dotenv import load_dotenv
from portia.runner import Runner
from portia.config import Config, StorageClass, LogLevel, LLMProvider, LLMModel

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load the default config and override the storage class to point to the Portia cloud
my_config = Config.from_default(
    storage_class=StorageClass.CLOUD,
    default_log_level=LogLevel.DEBUG,
    llm_provider=LLMProvider.OPENAI, # You can use `MISTRAL`, `ANTHROPIC` instead
    llm_model=LLMModel.GPT_4_O, # You can use any of the available models instead
    openai_api_key=OPENAI_API_KEY # Use `mistralai_api_key=MISTRALAI` or `anthropic_api_key=ANTHROPIC_API_KEY` instead
)

# Instantiate a Portia runner. Load it with the default config and with the simple tool above.
runner = Runner(config=my_config, tool_registry=example_tool_registry)

# Execute a workflow from the user query
workflow = runner.execute_query('Which stock price grew faster in 2024, Amazon or Google?')

# Serialise into JSON an print the output
print(workflow.model_dump_json(indent=2))
```

## Learn more
- Head over to our docs at <a href="docs.portialabs.ai" target="_blank">**docs.portialabs.ai (↗)**</a>.
- Join the conversation on our <a href="https://discord.gg/DvAJz9ffaR" target="_blank">**Discord channel (↗)**</a>.
- Watch us embarrass ourselves on our <a href="https://www.youtube.com/@PortiaAI" target="_blank">**YouTube channel (↗)**</a>.
- Follow us on <a href="https://www.producthunt.com/posts/portia-ai" target="_blank">**Product Hunt (↗)**</a>.

## Contribution guidelines
Head on over to our <a href="https://github.com/portiaAI/portia-sdk-python/blob/main/CONTRIBUTING.md" target="_blank">**contribution guide (↗)**</a> for details.



---
HERE BE TOM STUFF
---


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
        logger().debug(f"Adding {a} and {b}")
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
