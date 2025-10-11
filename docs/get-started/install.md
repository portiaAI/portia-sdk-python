* [Get started](/)
* Install and setup

On this page

# Install and setup

Let's get you set up and run a test query to make sure everything is in order.

Requirements

Portia requires **python v3.11 and above**. If you need to update your python version please visit their [docs](https://python.org/downloads/). If you are unsure what python version you have, you can check using

```
python3 --version
```

### Install the Portia Python SDK[​](#install-the-portia-python-sdk "Direct link to Install the Portia Python SDK")

Run the following command to install our SDK and its dependencies. The command below assumes you're using **pip** as your installer. For poetry the install command would be `poetry add ...` and for uv it's `uv add ...`. The args otherwise remain the same.

```
pip install portia-sdk-python
```

Out of the box the SDK comes with dependencies for OpenAI (and Azure OpenAI) + Anthropic. We additionally support Amazon Bedrock, Mistral and Google GenAI (Gemini). These dependencies can be added with:

```
pip install "portia-sdk-python[all]"  
# Or only with Amazon Bedrock extra dependencies  
pip install "portia-sdk-python[amazon]"  
# Or only with Google GenAI extra dependencies  
pip install "portia-sdk-python[google]"  
# Or only with Mistral extra dependencies  
pip install "portia-sdk-python[mistral]"
```

### Configure access to your preferred LLM[​](#configure-access-to-your-preferred-llm "Direct link to Configure access to your preferred LLM")

Set environment variables to connect to one of our currently supported LLMs. We are currently expanding this list. See [**Configure LLM options ↗**](/manage-config#configure-llm-options) for more information on how to configure Portia for different LLM providers and models.

* Open AI
* Anthropic
* Mistral
* Google GenAI
* Azure OpenAI
* Amazon Bedrock

`gpt-4.1` is set as the default model. You can sign up to their platform **[here](https://platform.openai.com/signup)**

```
export OPENAI_API_KEY='your-api-key-here'
```

`sonnet-3.7` and `sonnet-3.5 are both used by default. You can sign up to their platform **[here](https://www.anthropic.com/api)**

```
export ANTHROPIC_API_KEY='your-api-key-here'
```

`mistral-large-latest` is set as the default model. You can sign up to their platform **[here](https://auth.mistral.ai/ui/registration)**

Ensure Mistral dependencies are installed with `pip install "portia-sdk-python[mistral]"` or `"portia-sdk-python[all]"`

```
export MISTRAL_API_KEY='your-api-key-here'
```

`gemini-2.5-pro` and `gemini-2.5-flash` are both used by default. You can sign up to their platform **[here](https://ai.google.dev/)**

Ensure Google GenAI dependencies are installed with `pip install "portia-sdk-python[google]"` or `"portia-sdk-python[all]"`

```
export GOOGLE_API_KEY='your-api-key-here'
```

`gpt-4.1` is set as the default model. You can sign up to their platform **[here](https://azure.microsoft.com/en-us/products/ai-services/openai-service)**

```
export AZURE_OPENAI_API_KEY='your-api-key-here'  
export AZURE_OPENAI_ENDPOINT='your-api-key-here'
```

`eu.anthropic.claude-3-7-sonnet-20250219-v1:0` is set as the default model. You can sign up to their platform **[here](https://aws.amazon.com/bedrock/)**

Ensure Amazon dependencies are installed with `pip install "portia-sdk-python[amazon]"` or `"portia-sdk-python[all]"`

```
export AWS_ACCESS_KEY_ID = 'your-access-key-id'  
export AWS_SECRET_ACCESS_KEY = 'your-secret-access-key'  
export AWS_DEFAULT_REGION = 'your-default-region'  
# OR if you want using you ~/.aws/credentials  
export AWS_CREDENTIALS_PROFILE_NAME = 'your-credentials-profile-name'
```

### Test your installation from the command line[​](#test-your-installation-from-the-command-line "Direct link to Test your installation from the command line")

Let's submit a basic prompt to your LLM using our framework to make sure it's all working fine. We will submit a simple maths question, which should invoke one of the open source tools in our SDK:

* Open AI
* Anthropic
* Mistral
* Google GenAI
* Azure OpenAI
* Amazon Bedrock

Open AI is the default LLM provider. Just run:

```
portia-cli run "add 1 + 2"
```

To use Anthropic from the CLI, just run:

```
portia-cli run --llm-provider="anthropic" "add 1 + 2"
```

To use Mistral from the CLI, just run:

```
portia-cli run --llm-provider="mistralai" "add 1 + 2"
```

To use Google GenAI from the CLI, just run:

```
portia-cli run --llm-provider="google" "add 1 + 2"
```

To use Azure OpenAI from the CLI, just run:

```
portia-cli run --llm-provider="azure-openai" "add 1 + 2"
```

To use Amazon Bedrock models from the CLI, just run:

```
portia-cli run --llm-provider="amazon" "add 1 + 2"
```

Are you stuck? Try this 😅

Remember to use the **command specific to your installer**. The instructions above are for **pip** specifically. For other installers use one of the commands below (args don't change):

* For poetry, the run command is `poetry run portia-cli ...`.
* For uv, the run command is `uv run portia-cli ...`.

Make sure you're in the **right directory or venv** as well (where your lock file is)!

Portia will return the final state of the plan run created in response to the submitted prompt. We will delve into plan run states more deeply in a later section but for now you want to be sure you can see `"state": "COMPLETE"` and the answer to your maths question e.g. `"final_output": {"value": 3.0}` as part of that returned state. Here's an example output:

```
{  
    "id": "prun-13a97e70-2ca6-41c9-bc49-b7f84f6d3982",  
    "plan_id": "plan-96693022-598e-458c-8d2f-44ba51d4f0b5",  
    "current_step_index": 0,  
    "clarifications": [],  
    "state": "COMPLETE",  
    "step_outputs": {  
        "$result": {  
            "value": 3.0  
        }  
    },  
    "final_output": {  
        "value": 3.0  
    }  
}
```

### Test your installation from a Python file[​](#test-your-installation-from-a-python-file "Direct link to Test your installation from a Python file")

As a final verification step for your installation, set up the required environment variables in the `.env` of a project directory of your choice, namely the relevant LLM API keys. We can now replicate the CLI-driven test above from a python file within that directory.

* Open AI
* Anthropic
* Mistral
* Google GenAI
* Amazon Bedrock
* Azure OpenAI

In your local `.env` file, set up your API key as an environment variable using `OPENAI_API_KEY`.  
Then create a file e.g. `main.py` in your project directory and paste the following code in.

main.py

```
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    default_config,  
    example_tool_registry,  
)  
  
load_dotenv()  
  
# Instantiate Portia with the default config which uses Open AI, and with some example tools.  
portia = Portia(tools=example_tool_registry)  
# Run the test query and print the output!  
plan_run = portia.run('add 1 + 2')  
print(plan_run.model_dump_json(indent=2))
```

In your local `.env` file, set up your API key as an environment variable using `ANTHROPIC_API_KEY`.  
Then create a file e.g. `main.py` in your project directory and paste the following code in.

main.py

```
import os  
from dotenv import load_dotenv  
from portia import (  
    Config,  
    LLMProvider,  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')  
  
# Create a default Portia config with LLM provider set to Anthropic and to the Sonnet 3.5 model  
anthropic_config = Config.from_default(  
    llm_provider=LLMProvider.ANTHROPIC,  
    default_model="anthropic/claude-3-5-sonnet-latest",  
    anthropic_api_key=ANTHROPIC_API_KEY  
)  
# Instantiate a Portia instance. Load it with the config and with the example tools.  
portia = Portia(config=anthropic_config, tools=example_tool_registry)  
# Run the test query and print the output!  
plan_run = portia.run('add 1 + 2')  
print(plan_run.model_dump_json(indent=2))
```

In your local `.env` file, set up your API key as an environment variable using `MISTRAL_API_KEY`.  
Then create a file e.g. `main.py` in your project directory and paste the following code in.

main.py

```
import os  
from dotenv import load_dotenv  
from portia import (  
    Config,  
    LLMProvider,  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')  
  
# Create a default Portia config with LLM provider set to Mistral AI and the latest Mistral Large model  
mistral_config = Config.from_default(  
    llm_provider=LLMProvider.MISTRALAI,  
    default_model="mistralai/mistral-large-latest",  
    mistralai_api_key=MISTRAL_API_KEY  
)  
# Instantiate a Portia instance. Load it with the config and with the example tools.  
portia = Portia(config=mistral_config, tools=example_tool_registry)  
# Run the test query and print the output!  
plan_run = portia.run('add 1 + 2')  
print(plan_run.model_dump_json(indent=2))
```

In your local `.env` file, set up your API key as an environment variable using `GOOGLE_API_KEY`.  
Then create a file e.g. `main.py` in your project directory and paste the following code in.

main.py

```
import os  
from dotenv import load_dotenv  
from portia import (  
    Config,  
    LLMProvider,  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  
  
# Create a default Portia config with LLM provider set to Google GenAI and model set to Gemini 2.0 Flash  
google_config = Config.from_default(  
    llm_provider=LLMProvider.GOOGLE,  
    default_model="google/gemini-2.0-flash",  
    google_api_key=GOOGLE_API_KEY  
)  
# Instantiate a Portia instance. Load it with the config and with the example tools.  
portia = Portia(config=google_config, tools=example_tool_registry)  
# Run the test query and print the output!  
plan_run = portia.run('add 1 + 2')  
print(plan_run.model_dump_json(indent=2))
```

In your local `.env` file, set up your API key as an environment variable using `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION`.
You can also use the local aws credentials file e.g. `~/.aws/credentials` by just specifying the `aws_credentials_profile_name` parameter.

Then create a file e.g. `main.py` in your project directory and paste the following code in.

main.py

```
import os  
from dotenv import load_dotenv  
from portia import (  
    Config,  
    LLMProvider,  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')  
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')  
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')  
  
# Create a default Portia config ussing aws access keys with LLM provider set to AMAZON and model set to anthropic within Bedrock (make sure you enable the model in your Bedrock model access settings).  
amazon_config = Config.from_default(  
    llm_provider=LLMProvider.AMAZON,  
    default_model="amazon/eu.anthropic.claude-sonnet-4-20250514-v1:0",  
    aws_access_key_id=AWS_ACCESS_KEY_ID,  
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  
    aws_default_region=AWS_DEFAULT_REGION,  
)  
# Config using the aws_credentials_profile_name, if you're using ~/.aws/credentials (generated by `aws configure`).  
AWS_CREDENTIALS_PROFILE_NAME = os.getenv('AWS_CREDENTIALS_PROFILE_NAME') or "default"  
amazon_config2 = Config.from_default(  
    llm_provider=LLMProvider.AMAZON,  
    default_model="amazon/eu.anthropic.claude-sonnet-4-20250514-v1:0",  
    aws_credentials_profile_name=AWS_CREDENTIALS_PROFILE_NAME,  
)  
portia = Portia(config=amazon_config, tools=example_tool_registry)  
# Run the test query and print the output!  
plan_run = portia.run('add 1 + 2')  
print(plan_run.model_dump_json(indent=2))
```

In your local `.env` file, set up your API key and API endpoint as environment variables using `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`.  
Then create a file e.g. `main.py` in your project directory and paste the following code in.

main.py

```
import os  
from dotenv import load_dotenv  
from portia import (  
    Config,  
    LLMProvider,  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')  
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')  
  
# Create a default Portia config with LLM provider set to Azure OpenAI and model to GPT 4o  
azure_config = Config.from_default(  
    llm_provider=LLMProvider.AZURE_OPENAI,  
    default_model="azure-openai/gpt-4o",  
    azure_openai_api_key=AZURE_OPENAI_API_KEY,  
    azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,  
)  
# Instantiate a Portia instance. Load it with the config and with the example tools.  
portia = Portia(config=azure_config, tools=example_tool_registry)  
# Run the test query and print the output!  
plan_run = portia.run('add 1 + 2')  
print(plan_run.model_dump_json(indent=2))
```

You should see a similar output to the the CLI-driven test we ran in step 4.

We will review the various elements in `main.py` in more detail in later sections. For now you should remember that:

* You will use a `Portia` instance to handle user prompts.
* A `Portia` instance expects a `Config`. This is where you can specify things like the model you want to use and where you want to store plan runs.
* A `Portia` instance also expects `tools`. This can be a list of tools, or a `ToolRegistry` (i.e a collection of tools you want to use).

If you got this far then we're off to the races 🐎. Let's get you set up with a Portia account so you can also use our cloud features.
Don't worry it comes with a free trial ([**Pricing page ↗**](https://www.portialabs.ai/pricing)) 😉

Last updated on **Sep 9, 2025** by **robbie-portia**