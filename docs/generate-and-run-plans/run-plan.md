* [Generate and run plans](/generate-and-run-plans)
* Run a plan

On this page

# Run a plan

Learn how to run a plan run from an existing plan or end-to-end.

TL;DR

* A plan run is (uncontroversially) a unique run of a plan. It is represented by the `PlanRun` class ([**SDK reference ↗**](/SDK/portia/plan_run)).
* The `PlanRun` object tracks the state of the plan run and is enriched as every step of the plan is completed.
* A plan run can be generated from a plan using the `run_plan` method.
* You can also plan a query response, then create and execute a plan run in one fell swoop using the `run` method of the `Portia` instance class ([**SDK reference ↗**](/SDK/portia/portia)).

## Overview of plan runs in Portia[​](#overview-of-plan-runs-in-portia "Direct link to Overview of plan runs in Portia")

Portia captures the state of a plan run at every step in an auditable way. This includes:

* A step index tracking at which step we are in the plan run.
* The actual plan run state e.g. NOT\_STARTED, IN\_PROGRESS, COMPLETE, READY\_TO\_RESUME or NEED\_CLARIFICATION.
* A list of step outputs that is populated throughout the plan run.

In a later section we will also see that a plan run state also tracks the list of instances where human input was solicited during plan run, known as `Clarification`.

Plan run states are captured in the `PlanRun` class ([**SDK reference ↗**](/SDK/portia/plan_run)). In the previous section ([**Generate a plan ↗**](/generate-plan)), we generated a plan in response to the query `Which stock price grew faster in 2024, Amazon or Google?`. Let's examine the final state once we run that plan:

* Generated plan
* Plan run in final state

plan-1dcd74a4-0af5-490a-a7d0-0df4fd983977.json

```
{  
  "id": "plan-1dcd74a4-0af5-490a-a7d0-0df4fd983977",  
  "plan_context": {  
    "query": "Which stock price grew faster, Amazon or Google?",  
    "tool_ids": [  
      "calculator_tool",  
      "weather_tool",  
      "search_tool"  
    ]  
  },  
  "steps": [  
    {  
      "task": "Search for the latest stock price growth data for Amazon.",  
      "inputs": [],  
      "tool_id": "search_tool",  
      "output": "$amazon_stock_growth"  
    },  
    {  
      "task": "Search for the latest stock price growth data for Google.",  
      "inputs": [],  
      "tool_id": "search_tool",  
      "output": "$google_stock_growth"  
    },  
    {  
      "task": "Compare the stock price growth of Amazon and Google.",  
      "inputs": [  
        {  
          "name": "$amazon_stock_growth",  
          "description": "The stock price growth data for Amazon."  
        },  
        {  
          "name": "$google_stock_growth",  
          "description": "The stock price growth data for Google."  
        }  
      ],  
      "tool_id": "llm_tool",  
      "output": "$stock_growth_comparison"  
    }  
  ]  
}
```

prun-18d9aa91-0066-413f-af32-b979bce89821.json

```
{  
  "id": "prun-18d9aa91-0066-413f-af32-b979bce89821",  
  "plan_id": "plan-a89efeb0-51ef-4f2c-b435-a936c27c3cfc",  
  "current_step_index": 2,  
  "state": "COMPLETE",  
  "outputs": {  
    "clarifications": [],  
    "step_outputs": {  
      "$amazon_stock_growth": {  
        "value": "Amazon stock closed at an all-time high of $214.10 in November...",  
        "summary": null  
      },  
      "$google_stock_growth": {  
        "value": "In 2024, Google's parent company Alphabet surged 35.5% according to...",  
        "summary": null  
      },  
      "$faster_growth": {  
        "value": "In 2024, Amazon's stock price grew by 52%, while Google's parent company Alphabet saw a stock price surge of 35.5%.",  
        "summary": null  
      }  
    },  
    "final_output": {  
      "value": "In 2024, Amazon's stock price grew by 52%, while Google's parent company Alphabet saw a stock price surge of 35.5%.",  
      "summary": null  
    }  
  }  
}
```

Every plan run has a unique `id` and relates to a unique `plan_id`. If you were to attempt running the same plan multiple times, you would generate multiple `PlanRun` objects each with a unique `id` but all with the same `plan_id` property.

## Plan run state changes[​](#plan-run-state-changes "Direct link to Plan run state changes")

As Portia cycles through a plan run, each executed step produces an output. The plan run state is enriched with step outputs at every step of the execution as well. Note that in this example the main tool used is the 'Search Tool' provided in this SDK in the `example_tool_registry`, and wraps around the Tavily API. We will discuss tools in more depth in the next section.
You should be able to inspect the state changes for the above plan run in the logs when you run the code.

Animation above made on the brilliant [**snappify.com ↗**](https://snappify.com).

## Run from a pre-expressed plan[​](#run-from-a-pre-expressed-plan "Direct link to Run from a pre-expressed plan")

**Tavily API key required**

We will use a simple GET endpoint from Tavily in this section. Please sign up to obtain an API key from them ([**↗**](https://tavily.com/)) and set it in the environment variable `TAVILY_API_KEY`.

To get to an output that looks like the plan run example above, let's write some code to create a plan and then execute a plan run from that plan. Here is the code to do that:

* Sync
* Async

main.py

```
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    PlanBuilderV2,  
    StepOutput,  
    example_tool_registry,  
)  
  
load_dotenv()  
  
# Instantiate a Portia instance. Load it with the default config and with the example tools.  
portia = Portia(tools=example_tool_registry)  
  
# Build the plan  
plan = (  
    PlanBuilderV2()  
    .single_tool_agent_step(  
        step_name="amazon_stock_price",  
        task="Find the growth of Amazon's stock price since the start of 2024",  
        tool="search_tool",  
    )  
    .single_tool_agent_step(  
        step_name="google_stock_price",  
        task="Find the growth of Google's stock price since the start of 2024",  
        tool="search_tool",  
    )  
    .llm_step(  
        task="Determine which company has grown more since the start of 2024",  
        inputs=[StepOutput("amazon_stock_price"), StepOutput("google_stock_price")],  
    )  
    .build()  
)  
  
# Run the plan  
plan_run = portia.run_plan(plan)  
  
# Serialise into JSON and print the output  
print(plan_run.model_dump_json(indent=2))
```

main.py

```
import asyncio  
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    PlanBuilderV2,  
    StepOutput,  
    example_tool_registry,  
)  
  
load_dotenv()  
  
# Instantiate a Portia instance. Load it with the default config and with the example tools.  
portia = Portia(tools=example_tool_registry)  
  
async def main():  
    # Build the plan  
    plan = (  
      PlanBuilderV2()  
      .single_tool_agent_step(  
          step_name="amazon_stock_price",  
          task="Find the growth of Amazon's stock price since the start of 2024",  
          tool="search_tool",  
      )  
      .single_tool_agent_step(  
          step_name="google_stock_price",  
          task="Find the growth of Google's stock price since the start of 2024",  
          tool="search_tool",  
      )  
      .llm_step(  
          task="Determine which company has grown more since the start of 2024",  
          inputs=[StepOutput("amazon_stock_price"), StepOutput("google_stock_price")],  
      )  
      .build()  
  )  
  
    # Run the plan  
    plan_run = await portia.arun_plan(plan)  
  
    # Serialise into JSON and print the output  
    print(plan_run.model_dump_json(indent=2))  
  
# Run the async function  
asyncio.run(main())
```

Here we are storing the `Plan` object we created, then using the `run_plan` method to start a `PlanRun`.

info

If you want to see an example where a user iterates on a plan before we proceed with plan run, take a look at the intro example in our [**examples repo (↗)**](https://github.com/portiaAI/portia-agent-examples/blob/main/get_started_google_tools/README.md).

## Run directly from a user query[​](#run-directly-from-a-user-query "Direct link to Run directly from a user query")

**Tavily API key required**

We will use a simple GET endpoint from Tavily in this section. Please sign up to obtain an API key from them ([**↗**](https://tavily.com/)) and set it in the environment variable `TAVILY_API_KEY`.

You can also run a plan immediately from the user query, without examining the `Plan` object in between. This would generate a plan as an intermediate step as well but will also immediately spawn a plan run from it. You would simply use the `run` method from your `Portia` instance class like so:

* Sync
* Async

main.py

```
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
  
# Instantiate a Portia instance. Load it with the default config and with the example tools.  
portia = Portia(tools=example_tool_registry)  
  
# Generate the plan from the user query  
plan_run = portia.run('Which stock price grew faster in 2024, Amazon or Google?')  
  
# Serialise into JSON and print the output  
print(plan_run.model_dump_json(indent=2))
```

main.py

```
import asyncio  
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    example_tool_registry,  
)  
  
load_dotenv()  
  
# Instantiate a Portia instance. Load it with the default config and with the example tools.  
portia = Portia(tools=example_tool_registry)  
  
async def main():  
    # Generate the plan from the user query  
    plan_run = await portia.arun('Which stock price grew faster in 2024, Amazon or Google?')  
  
    # Serialise into JSON and print the output  
    print(plan_run.model_dump_json(indent=2))  
  
# Run the async function  
asyncio.run(main())
```

Track plan run states in logs

You can track plan run state changes live as they occur through the logs by setting `default_log_level` to DEBUG in the `Config` of your `Portia` instance ([**Manage logging ↗**](/manage-config#manage-logging)).

Last updated on **Sep 9, 2025** by **robbie-portia**