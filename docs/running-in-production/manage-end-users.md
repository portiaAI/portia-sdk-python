* [Running in production](/running-in-production)
* Managing end users

On this page

# Managing end users

Whilst building an agent for yourself can be very rewarding most agentic use cases run for many users. For example you may be an engineer creating a new agent for all the staff in your business. It may be important for the agent to know information about the specific member of staff that the agent is running for. Imagine a query like "Send me a summary of the latest results". This requires information about who the "me" is.

Portia has been built from the ground up for production deployments and so has a first class representation of your users within Portia. We call these entities end users, the people or companies that you are running agentic workflows for.

TL;DR

The `EndUser` class can be used to represent your users within `Portia`.

* The `external_id` field in an `EndUser` object uniquely represents the end user in your system e.g. an internal ID or an email address.
* `names`, `emails` and `phone_numbers` can all be stored against this object. They can dynamically be updated in tools with changes made to `end_user` models being persisted in storage.
* `additional_data` can be used to pass user specific info that may be relevant to the response such as title and department.
* Authentication is tied to the end user you use when executing a `plan_run`. This allows us to re-use Oauth tokens (subject to your token retention policy) improving user experience.

Important

* If you don't provide an `end_user` the system will generate an `end_user` to represent you as the developer. This is useful if you're building a system with only one user. You'll see this represented as users with the prefix `portia::`.

## End users at Portia[​](#end-users-at-portia "Direct link to End users at Portia")

In Production, you will be running plans for many stakeholders including customers, employees and partners. You may want to pass information specific to these individuals when they submit a prompt and / or information specific to the current context they are operating in (e.g. the particular app they are using when they submit their prompt to initiate a plan run).

We refer to these "person" entities as **end users** and represent them through the `EndUser` model.

* You can pass either a string or a full `EndUser` to the plan + run endpoints. The string or external ID can be any value that uniquely represents the end user in your system e.g. an internal ID or an email address.
* Alongside the `end_user_id` you can also provide a set of additional attributes in the `additional_data` field.

## Pass the EndUser to the plan run[​](#pass-the-enduser-to-the-plan-run "Direct link to Pass the EndUser to the plan run")

main.py

```
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    default_config,  
    example_tool_registry,  
)  
from portia.end_user import EndUser  
  
load_dotenv()  
  
portia = Portia(tools=example_tool_registry)  
  
# We can provide it as a string  
plan_run = portia.run(  
    "Get the temperature in Svalbard and write me a personalized greeting with the result.",  
    end_user="my_user_id_123"  
)  
  
# Or provide additional information through the model:  
plan_run = portia.run(  
    "Get the temperature in Svalbard and write me a personalized greeting with the result.",  
    end_user=EndUser(external_id="my_user_id_123", name="Nicholas of Patara")  
)  
  
print(plan_run.model_dump_json(indent=2))
```

The result of this code block will be the addition of an `end_user_id` within the `PlanRun` state, and a `final_output` that is indeed personalised to Saint Nicholas (known by his stage name Santa Claus):

plan\_run\_state.json

```
{  
  "id": "prun-d9991518-92d7-447f-bf28-4f7b9b8110ce",  
  "plan_id": "plan-4f497c60-c33e-40ea-95b4-cd2054559fff",  
  "current_step_index": 1,  
  "clarifications": [],  
  "state": "COMPLETE",  
  "end_user_id":  "DemoUser123",  
  "step_outputs": {  
    "$svalbard_temperature": {  
      "value": "The current weather in Svalbard is light snow with a temperature of -11.53°C."  
    },  
    "$personalized_greeting": {  
      "value": "Hello Nicholas of Patara, I hope you are keeping warm. With the current weather in Svalbard showing light snow and a temperature of -11.53°C, make sure to bundle up and stay cozy!"  
    }  
  },  
  "final_output": {  
    "value": "Hello Nicholas of Patara, I hope you are keeping warm. With the current weather in Svalbard showing light snow and a temperature of -11.53°C, make sure to bundle up and stay cozy!"  
  }  
}
```

## Accessing end users in a tool[​](#accessing-end-users-in-a-tool "Direct link to Accessing end users in a tool")

End User objects are passed through to the tool run function as part of the `ToolRunContext`. This allows you to access attributes for your users in tools.

You can also update attributes in tools, which will be persisted to storage upon completion of the tool call. This provides a way of storing useful data about the user.

main.py

```
from pydantic import BaseModel, Field  
from portia.tool import Tool, ToolRunContext  
  
class EndUserUpdateToolSchema(BaseModel):  
    """Input for EndUserUpdateTool."""  
  
    name: str | None = Field(default=None, description="The new name for the end user.")  
  
  
class EndUserUpdateTool(Tool):  
    """Updates the name of the plan runs end user."""  
  
    id: str = "end_user_update"  
    name: str = "End User Update Tool"  
    description: str = "Updates the name of the end user"  
    args_schema: type[BaseModel] = EndUserUpdateToolSchema  
    output_schema: tuple[str, str] = ("str", "str: The new name")  
  
    def run(self, ctx: ToolRunContext, name: str) -> str:  
        """Change the name."""  
        ctx.end_user.name = name  
        ctx.end_user.set_attribute("has_name_update", "true")  
        return name
```

## End user state management[​](#end-user-state-management "Direct link to End user state management")

As we mentioned above End Users are first class citizens in the Portia Ecosystem. This means they are independent entities with their own state. Changes you make to them are persisted in storage and we refresh the state before commencing `plan_runs`.

This is particularly relevant for the `additional_data` field on the End User. This field allows you to store any additional data you like against users. This can either be done through the cloud interface, by providing it when running a plan, or by updating it in a tool.

main.py

```
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    default_config,  
    example_tool_registry,  
)  
from portia.end_user import EndUser  
  
load_dotenv()  
  
portia = Portia(tools=example_tool_registry)  
  
plan_run = portia.run(  
    "Get the temperature in Svalbard and write me a personalized greeting with the result.",  
    end_user=EndUser(external_id="my_user_id_123", name="Nicholas of Patara", additional_data={"weather_preferences": "I prefer my weather in the form of a Haiku"})  
)
```

## End user and OAuth tokens[​](#end-user-and-oauth-tokens "Direct link to End user and OAuth tokens")

If you are using Portia Cloud Tools which support user level OAuth tokens, these tokens are stored against the EndUser of the `plan_run`. If you have the setting enabled (see Security), tokens will be reused for each end user reducing the number of authentication flows they must do.
This makes setting an `end_user` correctly important in this case to avoid token collision issues.

Last updated on **Sep 9, 2025** by **github-actions[bot]**