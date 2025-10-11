* [Generate and run plans](/generate-and-run-plans)
* Generate a plan

On this page

# Generate a plan

Learn how to create structured, multi-agent plans using your LLM of choice and familiarise yourself with the structure of plans created using Portia.

TL;DR

* A plan is the set of steps an LLM thinks it should take in order to respond to a user prompt.
* A plan is represented by the `PlanV2`/`Plan` class. They can be created in code using the `PlanBuilderV2` ([**SDK reference ↗**](SDK/portia/builder/plan_builder_v2)) class or generated from a user prompt using the `run` method of the `Portia` class ([**SDK reference ↗**](/SDK/portia/portia)).
* Portia uses optimised system prompts and structured outputs to ensure adherence to a plan.
* You can create your own plans manually or reload existing plans, which is especially useful for repeatable plan runs.

## Overview of plans in Portia[​](#overview-of-plans-in-portia "Direct link to Overview of plans in Portia")

A plan is the set of steps an LLM thinks it should take in order to respond to a user prompt. Plans are:

* **Immutable**: Once a plan is generated, it cannot be altered. This is important for auditability.
* **Structured**: We use optimised system prompts to guide the LLM along a simple design language when generating a plan. This makes the plan format predictable and easy to process for the purposes of automation.
* **Human-readable**: Our planning language is in a simple, serialisable format. It is easy to render and present to users in a human readable front-end experience. This helps your users easily stay on top of your LLM's reasoning.

While Portia generates a plan in response to a user prompt and then runs it, you also have the option to [**create plans yourself manually↗**](/build-plan). This is especially suitable for your users' more repeatable routines or if you are latency sensitive.

## Introducing a Plan[​](#introducing-a-plan "Direct link to Introducing a Plan")

Let's bring this one to life by looking at an example plan below, created in response to the query `Search for the latest SpaceX news from the past 48 hours and if there are at least 3 articles, email Avrana (avrana@kern.ai) a summary of the top 3 developments with subject 'Latest SpaceX Updates'`.

plan.json

```
{  
  "steps": [  
    {  
      "task": "Search for the latest SpaceX news from the past 48 hours using the search tool.",  
      "inputs": [],  
      "tool_id": "search_tool",  
      "output": "$spacex_news_results"  
    },  
    {  
      "task": "Summarize the top 3 developments from the SpaceX news articles.",  
      "inputs": [  
        {  
          "name": "$spacex_news_results",  
          "description": "The list of SpaceX news articles returned by the search tool."  
        }  
      ],  
      "tool_id": "llm_tool",  
      "output": "$spacex_summary",  
      "condition": "if $spacex_news_results contains at least 3 articles"  
    },  
    {  
      "task": "Email Avrana (avrana@kern.ai) a summary of the top 3 SpaceX developments with the subject 'Latest SpaceX Updates'.",  
      "inputs": [  
        {  
          "name": "$spacex_summary",  
          "description": "The summary of the top 3 SpaceX developments."  
        }  
      ],  
      "tool_id": "portia:google:gmail:send_email",  
      "output": "$email_sent",  
      "condition": "if $spacex_news_results contains at least 3 articles"  
    }  
  ]  
}
```

A plan includes a series of steps defined by

* `"task"` A task describing the objective of that particular step.
* `"input"` The inputs required to achieve the step. Notice how the LLM is guided to weave the outputs of previous steps as inputs to the next ones where applicable e.g. `$spacex_news_results` coming out of the first step acts as an input to the second one.
* `"tool_id"` Any relevant tool needed for the completion of the step. Portia is able to filter for the relevant tools during the multi-shot plan generation process. As we will see later on in this tutorial you can specify the tool registries (directories) you want when handling a user prompt, including local / custom tools and ones provided by third parties. In this example we are referencing tools from Portia's cloud-hosted library, prefixed with `portia:`.
* `"output"` The step's final output. As mentioned above, every step output can be referenced in future steps. As we will see shortly, these outputs are serialised and saved in plan run state as it is being executed.
* `"condition"` An optional condition that's used to control the execution of the step. If the condition is not met, the step will be skipped. This condition will be evaluated by our introspection agent, with the context of the plan and plan run state.

## Creating a plan[​](#creating-a-plan "Direct link to Creating a plan")

There are two ways of creating a plan in Portia:

* From natural language: When using the `.run()` method with Portia, the Portia planning agent creates a plan for you which is then run for you.
* Using code: You can use our plan builder interface to create reliable, repeatable plans using code.
  More details are provided on this on the [**Build a plan manually ↗**](/SDK/portia/portia) page.

## User led learning[​](#user-led-learning "Direct link to User led learning")

Example plans can be used to bias the planning agent towards actions, tool use and behaviours, while also improving its ability to generate more complex plans. Broadly, the process for doing this with portia is 3 steps below

* "Like" plans saved to Portia Cloud from the dashboard to signal that they are patterns you want to reinforce.
* Pull "Liked" plans based on semantic similarity to the user intent in a query by using our freshly minted `portia.storage.get_similar_plans` method ([**SDK reference ↗**](/SDK/portia/storage#get_similar_plans)).
* Finally, ingest those similar plans as example plans in the Planning agent using the `portia.run` method's `example_plans` property ([**SDK reference ↗**](/SDK/portia/)).

For a deep dive into this feature and a practical example, check out our [**ULL blog post on example plans ↗**](https://blog.portialabs.ai/improve-planning-with-user-led-learning).

## Structured Output Schema[​](#structured-output-schema "Direct link to Structured Output Schema")

For some plans you might want to have a structured output at the end of a plan, for this we allow the ability to attach a structured output schema to the plan that the summarizer agent will attempt to coerce the results to. This is optional. To use, attach to the Plan object, and any Plan Runs that are created from this will attempt to use structured output for the final result, this can pull information from any point of the plan steps and is not just the final step. To attach a schema, you can do it through the `PlanBuilderV2` or the `run()` interfaces, as below.

* Sync
* Async

plan\_structured\_output.py

```
from pydantic import BaseModel  
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    PlanBuilderV2,  
    default_config,  
    example_tool_registry,  
)  
  
load_dotenv()  
portia = Portia(tools=example_tool_registry)  
  
# Final Output schema type to coerce to  
class FinalPlanOutput(BaseModel):  
    result: float # result here is an integer output from calculator tool, but will be converted to a float via structured output  
  
# Example via plan builder  
plan = (  
    PlanBuilderV2(  
        "Addition agent",  
    )  
    .single_tool_agent_step(task="Add 1 + 1", tool="calculator_tool")  
    .final_output(output_schema=FinalPlanOutput)  
    .build()  
)  
plan_run = portia.run_plan(plan)  
  
# Example via natural language  
plan_run = portia.run("Add 1 + 1", structured_output_schema=FinalPlanOutput)
```

plan\_structured\_output.py

```
import asyncio  
from pydantic import BaseModel  
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    default_config,  
    example_tool_registry,  
    PlanBuilderV2,  
)  
  
load_dotenv()  
portia = Portia(tools=example_tool_registry)  
  
# Final Output schema type to coerce to  
class FinalPlanOutput(BaseModel):  
    result: float # result here is an integer output from calculator tool, but will be converted to a float via structured output  
  
plan = (  
    PlanBuilderV2(  
        "Addition agent",  
    )  
    .single_tool_agent_step(task="Add 1 + 1", tool="calculator_tool")  
    .final_output(output_schema=FinalPlanOutput)  
    .build()  
)  
  
async def main():  
    # Example via plan builder  
    plan_run = portia.arun_plan(plan)  
  
    # Example via natural language  
    plan2 = await portia.arun("Add 1 + 1", structured_output_schema=FinalPlanOutput)  
    # other async code  
  
# Run the async function  
asyncio.run(main())
```

Run the plan as normal and the final output will be an instance of the attached schema.

Last updated on **Sep 9, 2025** by **robbie-portia**