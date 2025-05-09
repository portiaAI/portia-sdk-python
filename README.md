<p align="center">
  <img style="width: 200px; height: 178px" src="Logo_Portia_Stacked_Black.png" />
</p>

# Portia SDK Python

Portia AI is an open source developer framework for stateful, authenticated agentic workflows. The core product accessible in this repository is extensible with our complimentary cloud features which are aimed at making production deployments easier and faster.
Play around, break things and tell us how you're getting on in our <a href="https://discord.gg/DvAJz9ffaR" target="_blank">**Discord channel (↗)**</a>. Most importantly please be kind to your fellow humans (<a href="https://github.com/portiaAI/portia-sdk-python/blob/main/CODE_OF_CONDUCT.md" target="_blank" rel="noopener noreferrer">**Code of Conduct (↗)**</a>).

If you want to dive straight in with an example, dive into one of our examples in our **[Examples Repo (↗)](https://github.com/portiaAI/portia-agent-examples)**.

<p align="center"><strong>🌟 Star Portia AI to stay updated on new releases!</strong></p>

## Why Portia AI
| Problem | Portia's answer |
| ------- | --------------- |
| **Planning:** Many use cases require visibility into the LLM's reasoning, particularly for complex tasks requiring multiple steps and tools. LLMs also struggle picking the right tools as their tool set grows: a recurring limitation for production deployments | **Multi-agent plans:** Our open source, multi-shot prompter guides your LLM to produce a [`Plan`](https://docs.portialabs.ai/generate-plan) in response to a prompt, weaving the relevant tools, inputs and outputs for every step. |
| **Execution:** Tracking an LLM's progress mid-task is difficult, making it harder to intervene when guidance is needed. This is especially critical for enforcing company policies or correcting hallucinations (hello, missing arguments in tool calls!) | **Stateful PlanRuns:** Portia will spin up a multi-agent [`PlanRun`](https://docs.portialabs.ai/execute-workflow) to execute on generated plans and track their state throughout execution. Using our [`Clarification`](https://docs.portialabs.ai/manage-clarifications) abstraction you can define points where you want to take control of run execution e.g. to resolve missing information or multiple choice decisions. Portia serialises the run state, and you can manage its storage / retrieval yourself or use our cloud offering for simplicity. |
| **Authentication:** Existing solutions often disrupt the user experience with cumbersome authentication flows or require pre-emptive, full access to every tool—an approach that doesn't scale for multi-agent assistants. | **Extensible, authenticated tool calling:** Bring your own tools on our extensible [`Tool`](https://docs.portialabs.ai/extend-tool-definitions) abstraction, or use our growing plug and play authenticated [tool library](https://docs.portialabs.ai/run-portia-tools), which will include a number of popular SaaS providers over time (Google, Zendesk, Hubspot, Github etc.). All Portia tools feature just-in-time authentication with token refresh, offering security without compromising on user experience. |


## Quickstart

### Installation

0. Ensure you have python 3.11 or higher installed. If you need to update your python version please visit their [docs](https://www.python.org/downloads/).
```bash
python --version
```

1. Install the Portia Python SDK
```bash
pip install portia-sdk-python 
```

>[!NOTE]
> OpenAI and Anthropic dependencies are included by default. We also provide the following extras:<br/>
> * **MistralAI**: `portia-sdk-python[mistral]`
> * **Google Generative AI**: `portia-sdk-python[google]`
> 
> Alternatively you can add all dependencies with `portia-sdk-python[all]`

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

### E2E example repo
We have a repo that showcases some of our core concepts to get you started. It's available <a href="https://github.com/portiaAI/portia-agent-examples" target="_blank">**here (↗)**</a>. We recommend starting with the <a href="https://github.com/portiaAI/portia-agent-examples/tree/main/get_started_google_tools" target="_blank">**Google Tools example (↗)**</a> if you are brand new to Portia.

### E2E example with open source tools
This example is meant to get you familiar with a few of our core abstractions:
- A `Plan` is the set of steps an LLM thinks it should take in order to respond to a user prompt. They are immutable, structured and human-readable.
- A `PlanRun` is a unique instantiation of a `Plan`. The purpose of a `PlanRun` is to capture the state of a unique plan run at every step in an auditable way.
- `Portia` orchestrates plan generation and execution, including the creation, pausing and resumption of plan runs.

Before running the code below, make sure you have the following keys set as environment variables in your .env file:
- An OpenAI API key (or other LLM API key) set as `OPENAI_API_KEY=`
- A Tavily <a href="https://tavily.com/" target="_blank">(**↗**)</a> API key set as `TAVILY_API_KEY=`

```python
from dotenv import load_dotenv
from portia import Portia, default_config, example_tool_registry

load_dotenv()

# Instantiate a Portia client. Load it with the default config and with the example tools.
portia = Portia(config=default_config(), tools=example_tool_registry)

# Generate the plan from the user query
plan = portia.plan('Which stock price grew faster in 2024, Amazon or Google?')
print(plan.model_dump_json(indent=2))

# Create and execute the run from the generated plan
plan_run = portia.run_plan(plan)

# Serialise into JSON and print the output
print(plan_run.model_dump_json(indent=2))
```

### E2E example with Portia cloud storage
Our cloud offering will allow you to easily store and retrieve plans in the Portia cloud, access our library of cloud hosted tools, and use the Portia dashboard to view plan runs, clarifications and tool call logs. Head over to <a href="https://app.portialabs.ai" target="_blank">**app.portialabs.ai (↗)**</a> and get your Portia API key. You will need to set it as the env variable `PORTIA_API_KEY`.<br/>
Note that this example also requires the environment variables `OPENAI_API_KEY` (or ANTHROPIC or MISTRALAI if you're using either) and `TAVILY_API_KEY` as the [previous one](#e2e-example-with-open-source-tools).

The example below introduces **some** of the config options available with Portia AI (check out our <a href="https://docs.portialabs.ai/manage-config" target="_blank">**docs (↗)**</a> for more):
- The `storage_class` is set using the `StorageClass.CLOUD` ENUM. So long as your `PORTIA_API_KEY` is set, runs and tool calls will be logged and appear automatically in your Portia dashboard at <a href="https://app.portialabs.ai" target="_blank">**app.portialabs.ai (↗)**</a>.
- The `default_log_level` is set using the `LogLevel.DEBUG` ENUM to `DEBUG` so you can get some insight into the sausage factory in your terminal, including plan generation, run states, tool calls and outputs at every step 😅
- The `llm_provider` and `xxx_api_key` (varies depending on model provider chosen) are used to choose the specific LLM provider. In the example below we're using GPT 4o, but you can use Anthropic, Gemini and others!

Finally we also introduce the concept of a `tool_registry`, which is a flexible grouping of tools.

```python
import os
from dotenv import load_dotenv
from portia import (
    Portia,
    Config,
    StorageClass,
    LogLevel,
    LLMProvider,
    example_tool_registry,
)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load the default config and override the storage class to point to the Portia cloud
my_config = Config.from_default(
    storage_class=StorageClass.CLOUD,
    default_log_level=LogLevel.DEBUG,
    llm_provider=LLMProvider.OPENAI, # You can use `MISTRAL`, `ANTHROPIC` instead
    default_model="openai/gpt-4o", # You can pass any of the OpenAI models here instead
    openai_api_key=OPENAI_API_KEY # Use `mistralai_api_key=MISTRALAI` or `anthropic_api_key=ANTHROPIC_API_KEY` instead
)

# Instantiate a Portia client. Load it with the config and with the open source example tool registry
portia = Portia(config=my_config, tools=example_tool_registry)

# Execute query.
plan_run = portia.run('Which stock price grew faster in 2024, Amazon or Google?')

# Serialise into JSON an print the output
print(plan_run.model_dump_json(indent=2))
```

## Learn more
- Head over to our docs at <a href="https://docs.portialabs.ai" target="_blank">**docs.portialabs.ai (↗)**</a>.
- Join the conversation on our <a href="https://discord.gg/DvAJz9ffaR" target="_blank">**Discord channel (↗)**</a>.
- Watch us embarrass ourselves on our <a href="https://www.youtube.com/@PortiaAI" target="_blank">**YouTube channel (↗)**</a>.
- Follow us on <a href="https://www.producthunt.com/posts/portia-ai" target="_blank">**Product Hunt (↗)**</a>.

## Contribution guidelines
Head on over to our <a href="https://github.com/portiaAI/portia-sdk-python/blob/main/CONTRIBUTING.md" target="_blank">**contribution guide (↗)**</a> for details.

# ⭐ Support
You can support our work best by leaving a star!

![star](https://github.com/user-attachments/assets/8df5e1d9-a0d4-40b4-9c51-945841744050)

We love feedback and suggestions. Please join our <a href="https://discord.gg/DvAJz9ffaR" target="_blank">**Discord channel (↗)**</a> to chat with us.
