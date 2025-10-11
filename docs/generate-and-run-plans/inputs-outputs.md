* [Generate and run plans](/generate-and-run-plans)
* Inputs and Outputs

On this page

# Inputs and Outputs

Inputs and outputs are the core of any agentic workflow, and Portia provides a flexible way to define and use them. Inputs are managed via the plan input interface, while structured outputs are managed via the plan structured output interface in conjunction with Pydantic BaseModels.

## Plan Inputs[​](#plan-inputs "Direct link to Plan Inputs")

So far the starting point for all plan runs is a user query for a specific set of inputs e.g. "get the weather in Beirut". This is in contrast to a generalised query e.g. "get the weather for a given city" where the city is provided dynamically per plan run. The PlanInput abstraction allows you to use a generalised query or plan "template" where the input differs with every plan run.

In the planning stage, you define the list of plan inputs, providing a name and optional description for each, and pass them into the plan builder. This creates references that you can use later in your plan, using `Input("input_name")`. Then, when you run the plan, Portia expects you to provide specific values for the inputs at each run.

For example, consider a simple agent that tells you the weather in a particular city, with the city provided as a plan input.
To set this up, we define the plan input for the planner as follows:

* Sync
* Async

```
from portia import Portia, PlanBuilderV2, Input  
  
portia = Portia()  
  
# Specify the inputs you will use in the plan  
plan = (  
    PlanBuilderV2()  
    .input(name="city", description="The city to get the weather in")  
    .single_tool_agent_step(  
        task="Get the weather in the provided city",  
        tool="weather_tool",  
        inputs=[Input("city")],  
    ).build()  
)  
  
# This will create a single step plan that uses the weather tool with $city as an input to that tool.  
# Then, when running the plan, we pass in a value for the input. In this case, we select "London".  
# This value will then be used for the `$city` input in the plan and we will find the temperature in London.  
  
# Specify the values for those inputs when you run the plan  
plan_run_inputs = {"name": "city", "value": "London"}  
plan_run = portia.run_plan(plan, plan_run_inputs=[plan_run_inputs])
```

```
import asyncio  
from portia import Portia, PlanBuilderV2, Input  
  
portia = Portia()  
  
async def main():  
    # Specify the inputs you will use in the plan  
    plan = (  
      PlanBuilderV2()  
      .input(name="city", description="The city to get the weather in")  
      .single_tool_agent_step(  
          task="Get the weather in the provided city",  
          tool="weather_tool",  
          inputs=[Input("city")],  
      ).build()  
  )  
  
    # This will create a single step plan that uses the weather tool with $city as an input to that tool.  
    # Then, when running the plan, we pass in a value for the input. In this case, we select "London".  
    # This value will then be used for the `$city` input in the plan and we will find the temperature in London.  
  
    # Specify the values for those inputs when you run the plan  
    plan_run_inputs = {"name": "city", "value": "London"}  
    plan_run = await portia.arun_plan(plan, plan_run_inputs=[plan_run_inputs])  
  
# Run the async function  
asyncio.run(main())
```

## Plan Structured Outputs[​](#plan-structured-outputs "Direct link to Plan Structured Outputs")

For some plans you might want to have a structured output at the end of a plan, for this we allow the ability to attach a structured output schema to the plan that the summarizer agent will attempt to coerce the results to. This is optional and is based on [**Pydantic BaseModels ↗**](https://docs.pydantic.dev/latest/#pydantic-examples). To use, attach to the Plan object, and any Plan Runs that are created from this will attempt to use structured output for the final result. This can pull information from any point of the plan steps and is not just the final step. To attach a schema, you can do it through the PlanBuilder or the `run()` interfaces, as below.

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
  
# Final output schema type to coerce to  
class FinalPlanOutput(BaseModel):  
    result: float # result here is an integer output from calculator tool, but will be converted   
    # to a float via structured output you can also add other fields here, and they will be   
    # included in the output, as per any other Pydantic BaseModel  
  
# Example via plan builder  
plan = (  
    PlanBuilderV2()  
    .single_tool_agent_step(  
        task="Add 1 + 1",  
        tool="calculator_tool",  
    )  
    .final_output(output_schema=FinalPlanOutput)  
    .build()  
)  
plan_run = portia.arun_plan(plan)  
  
  
# Example via run interface  
plan_run2 = portia.run("Add 1 + 1", structured_output_schema=FinalPlanOutput)
```

plan\_structured\_output.py

```
import asyncio  
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
  
# Final output schema type to coerce to  
class FinalPlanOutput(BaseModel):  
    result: float # result here is an integer output from calculator tool, but will be converted   
    # to a float via structured output you can also add other fields here, and they will be   
    # included in the output, as per any other Pydantic BaseModel  
  
async def main():  
    plan = (  
      PlanBuilderV2()  
      .single_tool_agent_step(  
          task="Add 1 + 1",  
          tool="calculator_tool",  
      )  
      .final_output(output_schema=FinalPlanOutput)  
      .build()  
    )  
    plan_run = portia.arun_plan(plan)  
  
    # Example via plan interface  
    plan_run2 = await portia.arun("Add 1 + 1", structured_output_schema=FinalPlanOutput)  
  
# Run the async function  
asyncio.run(main())
```

Run the plan as normal and the final output will be an instance of the attached schema. It will be coerced to the type of the BaseModel provided and follows all the same rules as a pydantic model, including validation and description for fields.

## LLM Step Outputs[​](#llm-step-outputs "Direct link to LLM Step Outputs")

The LLM step allows structured outputs to be returned from the LLM, and these will be coerced to the type of the BaseModel provided. This follows all the same rules as a pydantic model, including validation and description for fields in the same way as the plan structured output above, but only for an LLM call within the plan.

llm\_tool\_output.py

```
from portia import Portia, config, PlanBuilderV2  
from portia.open_source_tools.llm_tool import LLMTool  
from portia.open_source_tools.weather import WeatherTool  
import dotenv  
from pydantic import BaseModel, Field  
  
# basics  
dotenv.load_dotenv(override=True)  
config = config.Config.from_default()  
  
# structured output schema  
class WeatherOutput(BaseModel):  
    temperature: float  
    description: str = Field(description="A description of the weather")  
  
portia = Portia(config)  
  
plan = PlanBuilderV2(  
).single_tool_agent_step(  
  task="get the weather in london", tool="weather_tool",  
).llm_step(  
  task="summarize the weather", output_schema=WeatherOutput  
).build()
```

## Browser Tool Outputs[​](#browser-tool-outputs "Direct link to Browser Tool Outputs")

The BrowserTool allows structured outputs to be returned from a browser tool call, and these will be coerced to the type of the basemodel provided and follows all the same rules as a pydantic model, including validation and description for fields in the same way as the plan structured output above, but only for a browser tool call within the plan.

browser\_tool\_output.py

```
from portia import Portia, config, PlanBuilderV2  
from portia.open_source_tools.browser_tool import BrowserTool  
import dotenv  
from pydantic import BaseModel, Field  
  
# basics  
dotenv.load_dotenv(override=True)  
  
config = config.Config.from_default()  
  
  
class Recipes(BaseModel):  
    recipe_names: list[str] = Field(description="List of recipe names found on the page")  
  
browsertool = BrowserTool(structured_output_schema=Recipes) # structured output schema attached  
tools = [browsertool]  
portia = Portia(config, tools=tools)  
  
plan = PlanBuilderV2(  
).single_tool_agent_step(  
    task="Get all the names of recipes on the frontpage of bbcgoodfood.com",  
    tool=browsertool.id  
).build()
```

Last updated on **Sep 9, 2025** by **robbie-portia**