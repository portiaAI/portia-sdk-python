* [Generate and run plans](/generate-and-run-plans)
* Build a plan

On this page

# Build a plan

If you prefer to explicitly define a plan step by step rather than rely on our planning agent, e.g. for established processes in your business, you can use the `PlanBuilderV2` interface. This requires outlining all the steps, inputs, outputs and tools for your agent manually.

The `PlanBuilderV2` offers methods to create each part of the plan iteratively:

* `.llm_step()` adds a step that sends a query to the underlying LLM
* `.invoke_tool_step()` adds a step that directly invokes a tool. Requires mapping of step outputs to tool arguments.
* `.single_tool_agent_step()` is similar to `.invoke_tool_step()` but an LLM call is made to map the inputs to the step to what the tool requires creating flexibility.
* `.react_agent_step()` adds a ReAct (i.e. Reasoning + Acting) agent that is capable of using multiple tools and calling them multiple times in a loop until a task is achieved.
* `.function_step()` is identical to `.invoke_tool_step()` but calls a Python function rather than a tool with an ID.
* `.if_()`, `.else_if_()`, `.else_()` and `.endif()` are used to add conditional branching to the plan.
* `.user_verify()` and `user_input()` allow interacting with the user (via [**clarifications ↗**](/understand-clarifications)) for their input and verification
* `.input()` and `.final_output()` allow specifying inputs into the plan and the format of the plan's output
* `.add_steps()` allows you to compose a plan by merging multiple smaller plans, while `.add_step()` allows you to add your own custom steps

## Example[​](#example "Direct link to Example")

plan\_builder.py

```
from portia import PlanBuilderV2, StepOutput, Input  
  
plan = (  
    PlanBuilderV2("Write a poem about the price of gold")  
    .input(name="purchase_quantity", description="The quantity of gold to purchase in ounces")  
    .input(name="currency", description="The currency to purchase the gold in", default_value="GBP")  
    .invoke_tool_step(  
        step_name="Search gold price",  
        tool="search_tool",  
        args={  
            "search_query": f"What is the price of gold per ounce in {Input('currency')}?",  
        },  
        output_schema=CommodityPriceWithCurrency,  
    )  
    .function_step(  
        function=lambda price_with_currency, purchase_quantity: (  
            price_with_currency.price * purchase_quantity  
        ),  
        args={  
            "price_with_currency": StepOutput("Search gold price"),  
            "purchase_quantity": Input("purchase_quantity"),  
        },  
    )  
    .llm_step(  
        task="Write a poem about the current price of gold",  
        inputs=[StepOutput(0), Input("currency")],  
    )  
    .single_tool_agent_step(  
        task="Send the poem to Robbie in an email at donotemail@portialabs.ai",  
        tool="portia:google:gmail:send_email",  
        inputs=[StepOutput(2)],  
    )  
    .final_output(  
        output_schema=FinalOutput,  
    )  
    .build()  
)
```

* Sync
* Async

```
portia.run_plan(plan, plan_run_inputs={"purchase_quantity": 100, "currency": "GBP"})
```

```
import asyncio  
  
async def main():  
    await portia.arun_plan(plan, plan_run_inputs={"purchase_quantity": 100, "currency": "GBP"})
```

You can also view [**this example ↗**](https://github.com/portiaAI/portia-sdk-python/blob/main/example_builder.py) for a more in-depth example.

## Available Step Types[​](#available-step-types "Direct link to Available Step Types")

### LLM step[​](#llm-step "Direct link to LLM step")

Use `.llm_step()` to add a step that directly queries the LLM tool:

```
builder.llm_step(  
    task="Analyze the given data and provide insights",  
    inputs=[StepOutput("previous_step")],  
    output_schema=AnalysisResult,  
    step_name="analyze_data"  
)
```

The `output_schema` is a Pydantic model that is used for the structured output.

### Invoke Tool step[​](#invoke-tool-step "Direct link to Invoke Tool step")

Use `.invoke_tool_step()` to add a step that directly invokes a tool:

```
builder.invoke_tool_step(  
    tool="portia:tavily::search",  
    args={"query": "latest news about AI"},  
    step_name="search_news"  
)
```

### Function step[​](#function-step "Direct link to Function step")

Use `.function_step()` to add a step that calls a function. This is useful for manipulating data from other steps using code, streaming updates on the plan as it is run or adding in guardrails.

```
def process_data(data):  
    return {"processed": data.upper()}  
  
builder.function_step(  
    function=process_data,  
    args={"data": StepOutput(0)},  
    step_name="process_raw_data"  
)
```

### Single Tool Agent step[​](#single-tool-agent-step "Direct link to Single Tool Agent step")

Use `.single_tool_agent_step()` to add a step that calls a tool using arguments that are worked out dynamically from the inputs:

```
builder.single_tool_agent_step(  
    tool="web_scraper",  
    task="Extract key information from the webpage provided",  
    inputs=[StepOutput("text_blob_with_url")],  
    step_name="scrape_webpage"  
)
```

### ReAct Tool Agent step[​](#react-tool-agent-step "Direct link to ReAct Tool Agent step")

Use `.react_agent_step()` to add a step that uses a ReAct agent (Reasoning + Acting) to complete a task.
The ReAct agent can use multiple tools and call them multiple times as needed to complete the task. This allows you to balance deterministic flows with more exploratory goal-based behaviour.

```
builder.react_agent_step(  
    tools=["search_tool", "weather_tool", "email_tool"],  
    task="Find the weather in the 3 most Southerly countries in Europe and email it to me",  
    step_name="scrape_webpage"  
)
```

## Conditionals[​](#conditionals "Direct link to Conditionals")

Use `.if_()` to start a conditional block for advanced control flow:

```
(  
    builder  
    .if_(  
        condition=lambda web_page: len(web_page) > 100_000,  
        args={  
            "web_page": StepOutput("scrape_webpage")  
        }  
    )  
    .llm_step(  
        task="Summarise the web page",  
        inputs=[StepOutput("scrape_webpage")],  
        step_name="summarise_webpage"  
    )  
    .endif()  
)
```

`if_()` takes a predicate (named `condition`), which can either be a function, or a natural language string. If it is a function, then the function will be run to return a boolean indicating whether the condition passed. If it is a natural language string, then an LLM will be used to determine whether the string is true or false.

`args` is a dictionary of arguments to pass to the predicate. Like other step types, you can pass references or values (see the [Inputs and Outputs](#inputs-and-outputs) section below for more details).

Also note that you need to add an endif() at the end of the flow to indicate the end of the conditional branch.

Alternative branches can be added to the conditional block using `.else_if_()` and `.else_()`:

```
(  
    builder  
    .if_(  
        condition=lambda web_page: len(web_page) > 100_000,  
        args={  
            "web_page": StepOutput("scrape_webpage")  
        }  
    )   # ...  
    .else_if_(  
        condition=lambda web_page: len(web_page) < 100,  
        args={  
            "web_page": StepOutput("scrape_webpage")  
        }  
    )  
    .function_step(  
        function=lambda: raise_exception("Web page is too short"),  
    )  
    .else_()  
    .function_step(  
        function=lambda: print("All good!"),  
    )  
    .endif()  
)
```

As mentioned, the condition can be a natural language string. Just write a statement that can be evaluated to true or false and pass the relevant context via the `args`.

```
(  
    builder  
    .if_(  
        condition="The web page is about large cats",  
        args={  
            "web_page": StepOutput("scrape_webpage")  
        }  
    )  
)
```

Conditional blocks can be nested to create *even* more complex control flow!

```
(  
    builder  
    .if_(  
        condition=lambda web_page: len(web_page) > 100_000,  
        args={  
            "web_page": StepOutput("scrape_webpage")  
        }  
    )  
    # Nested conditional block  
    .if_(  
        condition=lambda web_page: len(web_page) > 1_000_000,  
        args={  
            "web_page": StepOutput("scrape_webpage")  
        }  
    )  
    .function_step(  
        function=lambda: raise_exception("Web page is too long"),  
    )  
    .endif()  
    # ... back to the outer conditional block  
)
```

## User Interaction[​](#user-interaction "Direct link to User Interaction")

### User Verification[​](#user-verification "Direct link to User Verification")

Use `.user_verify()` when you want to pause plan execution to ask a user to confirm or reject the provided message.
The plan will only continue if they confirm. If the user rejects, the plan execution will stop with an error.
The user interaction is handled via clarifications - see [**Understand clarifications ↗**](/understand-clarifications) for more details.

```
builder.user_verify(  
    message=f"Do you want to proceed with the purchase? Price is {StepOutput('Calculate total price')}")
```

### User Input[​](#user-input "Direct link to User Input")

Use `.user_input()` when you want to pause plan execution for a user to provide input into the plan.
This input can either be in the form of free text or can be a multiple-choice set of options.
As with user verification, the user interaction is handled via clarifications - see [**Understand clarifications ↗**](/understand-clarifications) for more details.

```
# An example with multiple choice options  
builder.user_input(  
    message="How much would you like to purchase?",  
    options=[50, 100, 200],  
)  
  
# An example with a free text input  
builder.user_input(  
    message="Please enter your favourite food:",  
)
```

## Inputs and Outputs[​](#inputs-and-outputs "Direct link to Inputs and Outputs")

### Adding Plan Inputs[​](#adding-plan-inputs "Direct link to Adding Plan Inputs")

Use `.input()` to define inputs that the plan expects:

```
builder.input(  
    name="user_query",  
    description="The user's question or request"  
)
```

You can also provide the default value for the input, e.g

```
builder.input(  
    name="user_query",  
    description="The user's question or request",  
    # Default values can be overriden in plan_run_inputs but will be used as the fallback.  
    default_value="What is the capital of France?"  
)
```

You can dynamically add the value of the plan at run time, e.g

```
portia.run_plan(plan, plan_run_inputs={"user_query": "What is the capital of Peru?"})
```

You can also access nested fields of the input using the path attribute:

```
class UserProfile(BaseModel):  
    name: str  
    email: str  
  
class UserData(BaseModel):  
        profile: UserProfile  
        age: int  
  
builder.input(name="user_data").llm_step(task="Do some task", inputs=[Input("user_data", path="profile.name")])
```

### Referencing Step Outputs[​](#referencing-step-outputs "Direct link to Referencing Step Outputs")

You can reference outputs from previous steps using `StepOutput`:

```
from portia import StepOutput  
  
builder.invoke_tool_step(  
    tool="calculator",  
    args={"expression": f"This is some string {StepOutput('previous_step')} interpolation"}  
)
```

You can also reference previous step outputs using their index:

```
from portia import StepOutput  
  
builder.invoke_tool_step(  
    tool="calculator",  
    args={"expression": StepOutput(1)},  
)
```

As with Input, you can access nested fields of the output using the path attribute

```
# Access the .profile.name field of the output from the 'get_user_data' step  
builder.llm_step(task="Do some task", inputs=[StepOutput("get_user_data", path="profile.name")])
```

Note

The index of a step is the order in which it was added to the plan.
They are zero-indexed, so the first step is step 0.

Conditional clauses (`.if_()`, `.else_if_()`, `.else_()` and `.endif()`) *are* counted as steps and do have an index. Steps within a conditional branch are also counted - the step index is the order the steps appear in the plan, not the runtime index.

### Final Output Configuration[​](#final-output-configuration "Direct link to Final Output Configuration")

Use `.final_output()` to configure the final output:

```
plan = builder.final_output(  
    output_schema=FinalResult,  
    summarize=True  
).build()
```

* Sync
* Async

```
plan_run = portia.run_plan(plan)
```

```
import asyncio  
plan_run = asyncio.run(portia.arun_plan(plan))
```

# Will match `FinalResult` schema

final\_output\_value = plan\_run.outputs.final\_output.value

# Provides a succinct summary of the outputs (calls LLM to populate)

final\_output\_summary = plan\_run.outputs.final\_output.summary

```
## Building the Plan  
  
Once you've defined all your steps, call `.build()` to create the final plan:  
  
```python depends_on=builder_invisible_setup  
plan = builder.build()
```

The returned `PlanV2` object is ready to be executed with your Portia instance.

## [DEPRECATED] Build a plan manually[​](#deprecated-build-a-plan-manually "Direct link to [DEPRECATED] Build a plan manually")

Deprecation warning

There is an older form of the plan builder described below which is still functional in the SDK but over time we will be replacing it with PlanBuilderV2.

If you prefer to explicitly define a plan step by step rather than rely our planning agent, e.g. for established processes in your business, you can use the PlanBuilder interface. This obviously implies outlining all the steps, inputs, outputs and tools.

The `PlanBuilder` offers methods to create each part of the plan iteratively

* `.step` method adds a step to the end of the plan. It takes a `task`, `tool_id` and `output` name as arguments.
* `.input` and `.condition` methods add to the last step added, but can be overwritten with a `step_index` variable, and map outputs from one step to inputs of chosen (default last step), or considerations
* `.build` finally builds the `Plan` objective

plan\_builder.py

```
from portia.plan import PlanBuilder  
  
query = "What is the capital of france and what is the population of the city? If the city has a population of over 1 million, then find the mayor of the city."  
  
plan = PlanBuilder(  
  query # optional to provide, as the steps are built below, but provides context for storage and plan purpose  
).step(  
    task="Find the capital of france", # step task  
    tool_id="google_search", # tool id maps to a tool in the tool registry  
    output="$capital_of_france", # output variable name maps step output to variable  
).step(  
    task="Find the population of the capital of france",  
    tool_id="google_search",  
    output="$population_of_capital",  
).input( # add an input to step 2  
    name="$capital_of_france", # input variable name maps to a variable in the plan run outputs from step 1  
    description="Capital of france" # optional description for the variable  
).step(  
    task="Find the mayor of the city",  
    tool_id="google_search",  
    output="$mayor_of_city",  
).condition(  
    condition="$population_of_capital > 1000000", # adding a condition to the step  
).build() # build the plan once finalized
```

Last updated on **Sep 9, 2025** by **robbie-portia**