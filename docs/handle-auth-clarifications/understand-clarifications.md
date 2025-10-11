* [Handle auth and clarifications](/handle-auth-clarifications)
* Understand clarifications

On this page

# Understand clarifications

Define clarifications to bring structured input into a plan run.
Understand the different types of clarifications and how to use them.

TL;DR

* An agent can raise a clarification during a plan run to pause it and solicit human input. This pauses the plan run, serialises and saves its state at the step where clarification was raised.
* We represent a clarification with the `Clarification` class ([**SDK reference ↗**](/SDK/portia/clarification)). This includes useful information such as guidance to be surfaced to the user when soliciting their input. Because it is a structured object, you can easily serve it to an end user using a front end of your choosing when it is encountered.
* The user response is captured in the `Clarification` object itself, which is part of the `PlanRun` state. This means the plan run can be resumed, and the step at which the clarification was required can now be completed.

## Intro to clarifications[​](#intro-to-clarifications "Direct link to Intro to clarifications")

Portia introduces the concept of clarifications. An agent can raise a clarification during a plan run to pause it and solicit human input. This is important because:

1. LLM-driven tasks that are multi-step can be brittle and unreliable e.g. if an input is missing the LLM may hallucinate it. Instead we allow you to pause Portia-managed plan run and raise a clarification to the user so they can resolve the missing input for the LLM.
2. During plan run, there may be tasks where your organisation's policies require explicit approvals from specific people e.g. allowing bank transfers over a certain amount. Clarifications allow you to define these conditions so the agent running a particular step knows when to pause the plan run and solicit input in line with your policies.
3. More advanced use cases of clarifications also include hand off to a different part of your system based on certain conditions having been met. The structured nature of clarifications make this handoff easy to manage.

When Portia encounters a clarification and pauses a plan run, it serialises and saves the latest plan run state. Once the clarification is resolved, the obtained human input captured during clarification handling is added to the plan run state and the agent can resume step execution.

![Clarifications at work](/assets/images/clarifications_diagram-d8a0d674e55818e13ec3faae4c25c746.png)

## Types of clarifications[​](#types-of-clarifications "Direct link to Types of clarifications")

Clarifications are represented by the `Clarification` class ([**SDK reference ↗**](/SDK/portia/clarification)). Because it is a structured object, you can easily serve it to an end user using a front end of your choosing when it is encountered e.g. a chatbot or app like Slack, email etc.

We offer five categories of clarifications at the moment. You can see the properties and behaviours specific to each type in the tabs below. The common properties across all clarifications are:

* `uuid`: Unique ID for this clarification
* `category`: The type of clarification
* `response`: User's response to the clarification
* `step`: Plan run step where this clarification was raised
* `user_guidance`: Guidance provided to the user to explain the nature of the clarification
* `resolved`: Boolean of the clarification state

* Action clarifications
* Input clarifications
* Multiple choice clarifications
* Value confirmation clarifications
* Custom clarifications

Action clarifications are useful when a user action is needed to complete a step e.g. clicking on an `action_url` to complete an authentication flow or to make a payment. You will need to have a way to receive a callback from such a flow in order to confirm whether the clarification was resolved successfully.

action\_clarification.json

```
{  
    "uuid": "clar-425c8ce9-8fc9-43af-b99e-64903043c5df",  
    "plan_run_id": "prun-89c6bd4f-29d2-4aad-bf59-8ba3229fd258",  
    "category": “Action”,  
    "response": “success”,  
    "step": 1,  
    "user_guidance": "Click here to authenticate",  
    "resolved": true,  
    "action_url": “https://accounts.google.com/o/oauth2/…”,  
}
```

Input clarifications are used when a tool call is missing one argument and the user needs to provide it e.g. a `send_email` tool needs to be invoked but no email is resolvable from the user query. The `argument` attribute points to the tool argument this clarification resolves.

input\_clarification.json

```
{  
    "uuid": "clar-425c8ce9-8fc9-43af-b99e-64903043c5df",  
    "plan_run_id": "prun-89c6bd4f-29d2-4aad-bf59-8ba3229fd258",  
    "category": “Input”,  
    "response": “avrana@kern.ai”,  
    "step": 2,   
    "user_guidance": "Please provide me with Avrana's email address",   
    "resolved": true,  
    "argument": "$avrana_email",  
}
```

Multiple choice clarifications are raised when a tool argument is restricted to a list of values but the agent attempting to invoke the tool is given an argument that falls outside that list. The clarification can be used to serve the acceptable list of values for the user to choose from via the `options` attribute.

multiple\_choice\_clarification.json

```
{  
    "uuid": "clar-425c8ce9-8fc9-43af-b99e-64903043c5df",  
    "plan_run_id": "prun-89c6bd4f-29d2-4aad-bf59-8ba3229fd258",  
    "category": “Multiple Choice”,  
    "response": “ron_swanson@pawnee.com,  
    "step": 2,   
    "user_guidance": "Please select a recipient.",   
    "resolved": true,  
    "argument": "$recipient",  
    "options": [  
            "ron_swanson@pawnee.com",  
            "ron_burgundy@kvwnchannel4.com",  
            "ron@gone_wrong.com"  
        ]  
}
```

Value confirmation clarifications are raised to get the user to confirm or deny if they want to proceed with a particular value. This is particularly useful for 'human in the loop' tasks where you want to get the user to confirm the value before proceeding.

value\_confirmation\_clarification.json

```
{  
    "uuid": "clar-425c8ce9-8fc9-43af-b99e-64903043c5df",  
    "plan_run_id": "prun-89c6bd4f-29d2-4aad-bf59-8ba3229fd258",  
    "category": “Value Confirmation”,  
    "step": 2,   
    "user_guidance": "This will email all contacts in your database. Are you sure you want to proceed?",   
    "resolved": true,  
    "argument": "$email_all_contacts"  
}
```

Custom clarifications enable you to attach arbitrary information to a clarification.

custom\_clarification.json

```
{  
    "uuid": "clar-425c8ce9-8fc9-43af-b99e-64903043c5df",  
    "plan_run_id": "prun-89c6bd4f-29d2-4aad-bf59-8ba3229fd258",  
    "category": “Custom”,  
    "step": 2,   
    "user_guidance": "Which product did you want to buy?",   
    "resolved": true,  
    "data": {  
        "product_id": "prod-1234567890"  
    }  
}
```

## Clarification triggers[​](#clarification-triggers "Direct link to Clarification triggers")

Clarifications are raised in one of three scenarios:

1. LLM-triggered: During plan run, an agent attempting to complete a step notices that an input is missing, resulting in an Input clarification.
2. Tool-triggered: A clarification is explicitly raised in the python class definition of the tool in specific conditions e.g. if a requisite OAuth token is missing to complete the underlying API call or if a tool argument is invalid, resulting in Action or a Multiple Choice clarification respectively.
3. Starting or resuming a plan run: Before a plan run is started or resumed, the Portia runner checks the readiness of all tools mentioned in the plan. Clarifications are raised if any of the tools are not ready to be used. Portia tools use this mechanism to request user Authorization - see more details [**here ↗**](/run-portia-tools).

## Handle clarifications with your `Portia` instance[​](#handle-clarifications-with-your-portia-instance "Direct link to handle-clarifications-with-your-portia-instance")

Make a `weather.txt` file for this section

We're going to see how Portia handles multiple choices with clarifications. In this example we will import our open source tool `FileReaderTool` and ask it to open a non-existent local file `weather.txt`. This should trigger the tool to search for the file across the rest of the project directory and return all matches. Make sure to sprinkle a few copies of a `weather.txt` file around in the project directory.
Note: Our `weather.txt` file contains "The current weather in Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is broken clouds with a temperature of 6.76°C."

When the conditions requiring a clarification are met, the relevant tool call returns a `Clarification` object, the plan run is paused and the plan run state becomes `NEED CLARIFICATION`. Portia has now passed control of the plan run to you, the developer, along with the `Clarification` object in order for you to resolve with human or machine input. At this stage we need to make some changes in the `main.py` file to handle clarifications.

main.py

```
from portia import (  
    InMemoryToolRegistry,  
    MultipleChoiceClarification,  
    Portia,  
    PlanRunState,  
    default_config,  
)  
from portia.open_source_tools.local_file_reader_tool import FileReaderTool  
from portia.open_source_tools.local_file_writer_tool import FileWriterTool  
  
# Load open source tools into a tool registry. More on tool registries later in the docs!  
my_tool_registry = InMemoryToolRegistry.from_local_tools([FileReaderTool(), FileWriterTool()])  
# Instantiate a Portia instance. Load it with the default config and with the tools above  
portia = Portia(tools=my_tool_registry)  
  
# Execute the plan from the user query  
plan_run = portia.run('Read the contents of the file "weather.txt"')  
  
# Check if the plan run was paused due to raised clarifications  
while plan_run.state == PlanRunState.NEED_CLARIFICATION:  
    # If clarifications are needed, resolve them before resuming the plan run  
    for clarification in plan_run.get_outstanding_clarifications():  
        # For each clarification, prompt the user for input  
        print(f"{clarification.user_guidance}")  
        user_input = input("Please enter a value:\n" +  
                               (("\n".join(clarification.options) + "\n")   
                                if isinstance(clarification, MultipleChoiceClarification)  
                                else ""))  
        # Resolve the clarification with the user input  
        plan_run = portia.resolve_clarification(clarification, user_input, plan_run)  
  
    # Once clarifications are resolved, resume the plan run  
    plan_run = portia.resume(plan_run)  
  
# Serialise into JSON and print the output  
print(plan_run.model_dump_json(indent=2))
```

The keen eye may have noticed that we introduced the `InMemoryToolRegistry` class. Fear not, we will discuss tool registries in a later section. For now let's focus on the clarification handling sections highlighted in the code.
Remember to make sure you don't have a `weather.txt` file in the same folder as your python file AND make a few copies of a `weather.txt` file sprinkled around in other folders of the project directory. This will ensure that the prompt triggers the multiple choice clarifications on the `filename` argument of the `FileReaderTool`. The tool call will return a `Clarification` object per changes made in the previous section and pause the plan\_run.

The changes you need to make to our `main.py` in order to enable this behaviour are as follows:

1. Check if the state of the `PlanRun` object returned by the `run` method is `PlanRunState.NEED_CLARIFICATION`. This means the plan run paused before completion due to a clarification.
2. Use the `get_outstanding_clarifications` method of the `PlanRun` object to access all clarifications where `resolved` is false.
3. For each `Clarification`, surface the `user_guidance` to the relevant user and collect their input.
4. Use the `portia.resolve_clarification` method to capture the user input in the `response` attribute of the relevant clarification. Because clarifications are part of the plan run state itself, this means that the plan run now captures the latest human input gathered and can be resumed with the new information.
5. Once this is done you can resume the plan run using the `resume` method. In fact `resume` can take a `PlanRun` in any state as a parameter and will kick off that plan run from that current state. In this particular example, it resumes the plan run from the step where the clarifications were encountered.

For the example query above `Read the contents of the file "weather.txt".`, where the user resolves the clarification by entering one of the options offered by the clarification (in this particular case `demo_runs/weather.txt` in our project directory `momo_sdk_tests`), you should see the following plan run state and notice:

* The multiple choice clarification where the `user_guidance` was generated by Portia based on your clarification definition in the `FileReaderTool` class,
* The `response` in the second plan run snapshot reflecting the user input, and the change in `resolved` to `true` as a result
* The plan run `state` will appear to `NEED_CLARIFICATION` if you look at the logs at the point when the clarification is raised. It then progresses to `COMPLETE` once you respond to the clarification and the plan run is able to resume:

run\_state.json

```
{  
  "id": "prun-54d157fe-4b99-4dbb-a917-8fd8852df63d",  
  "plan_id": "plan-b87de5ac-41d9-4722-8baa-8015327511db",  
  "current_step_index": 0,  
  "state": "COMPLETE",  
  "outputs": {  
    "clarifications": [  
      {  
        "id": "clar-216c13a1-8342-41ca-99e5-59394cbc7008",  
        "category": "Multiple Choice",  
        "response": "../momo_sdk_tests/demo_runs/weather.txt",  
        "step": 0,  
        "user_guidance": "Found weather.txt in these location(s). Pick one to continue:\n['../momo_sdk_tests/demo_runs/weather.txt', '../momo_sdk_tests/my_custom_tools/__pycache__/weather.txt']",  
        "resolved": true,  
        "argument_name": "filename",  
        "options": [  
          "../momo_sdk_tests/demo_runs/weather.txt",  
          "../momo_sdk_tests/my_custom_tools/__pycache__/weather.txt"  
        ]  
      }  
    ],  
    "step_outputs": {  
      "$file_contents": {  
        "value": "The current weather in Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is broken clouds with a temperature of 6.76°C.",  
        "summary": null  
      }  
    },  
    "final_output": {  
      "value": "The current weather in Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is broken clouds with a temperature of 6.76°C.",  
      "summary": null  
    }  
  }  
}
```

## Handle clarifications with a `ClarificationHandler`[​](#handle-clarifications-with-a-clarificationhandler "Direct link to handle-clarifications-with-a-clarificationhandler")

Through the above example, we explicitly handle the clarifications in order to demonstrate the full clarification handling flow.
However, Portia also offers a `ClarificationHandler` class that can be used to simplify the handling of clarifications.
In order to use this, simply create your own class that inherits from `ClarificationHandler` and implement the methods for
handling the types of clarifications you expect to handle. Each method takes an `on_resolution` and `on_error` parameter -
these can be called either synchronously or asynchronously when the clarification handling is finished. This allows handling
clarifications in many different ways - for example, they could be handled by the user in a UI, or they could be handled in an
email or slack message.

Once you've created your clarifiication handler, it can be passed in as an execution hook when creating the Portia instance:

```
from portia import Clarification, ClarificationHandler, Config, ExecutionHooks, InputClarification, Portia  
from typing import Callable  
  
class CLIClarificationHandler(ClarificationHandler):  
    """Handles clarifications by obtaining user input from the CLI."""  
  
    def handle_input_clarification(  
        self,  
        clarification: InputClarification,  
        on_resolution: Callable[[Clarification, object], None],  
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002  
    ) -> None:  
        """Handle a user input clarifications by asking the user for input from the CLI."""  
        user_input = input(f"{clarification.user_guidance}\nPlease enter a value:\n")  
        on_resolution(clarification, user_input)  
  
portia = Portia(execution_hooks=ExecutionHooks(clarification_handler=CLIClarificationHandler()))
```

Portia also offers some default clarification handling behaviours that can be used out of the box. For example, you don't actually need
to implement your own CLI clarification handler (as done above) because our default CLI execution hooks, `CLIExecutionHooks`, provide a
clarification handler that allows the user to handle clarifications via the CLI.

```
from portia import Config, Portia  
from portia.cli import CLIExecutionHooks  
  
portia = Portia(execution_hooks=CLIExecutionHooks())
```

Last updated on **Sep 9, 2025** by **github-actions[bot]**