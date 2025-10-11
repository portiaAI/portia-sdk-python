* [Handle auth and clarifications](/handle-auth-clarifications)
* Run Portia tools with authentication

On this page

# Run Portia tools with authentication

Use clarifications to leverage Portia tools' native authentication support.

TL;DR

* All Portia tools come with built-in authentication, typically using Portia OAuth clients for each relevant resource server.
* At the start of a plan run containing Portia tools, `Portia` raises `ActionClarification`s to request user authorization for the subset of tools that require it.

Portia offers a cloud-hosted library of tools to save you development time. You can find the ever-growing list of Portia tools in the next section ([**Portia tool catalogue ↗**](/portia-tools)). All Portia tools come with plug and play authentication. Let's delve into how to handle the user authentication flow.

## Handling auth with `Clarification`[​](#handling-auth-with-clarification "Direct link to handling-auth-with-clarification")

We established in the preceding section that clarifications are raised when an agent needs input to progress. This concept lends itself perfectly to tool authentication. Let's break it down:

* All Portia tools come with built-in authentication, typically using Portia OAuth clients for each relevant resource server.
* Portia provisions the required token with the relevant scope when a tool call needs to be made.
* Tokens provisioned by Portia have a user-configurable retention period (see more [**here ↗**](/security)). The tokens are scoped to the `end_user` that was passed when running the plan or a default `end_user` if none was provided. You will need to reuse this `end_user_id` across plan runs to leverage token reusability ([**Manage multiple end users ↗**](/manage-end-users)).
* When a plan is run, before we start executing the steps, we first check the readiness of all tools contained in the steps of the plan. For Portia tools supporting OAuth, readiness includes validating that we have an access token stored for the `end_user_id` provided. If no OAuth token is found, an `ActionClarification` is raised with an OAuth link as the action URL. This OAuth link uses the relevant Portia authentication client and a Portia redirect URL.
* Portia's OAuth server listens for the authentication result and resolves the concerned clarification, allowing the plan run to resume again.

info

Note that there may be multiple tools that require permissions from the same OAuth client. In this case, Portia will combine together the required scopes, reducing the number of `ActionClarification`s that need to be resolved.

Optionally, you can configure a custom URL where users will be redirected after successful authentication. To do so, follow these steps:

1. Log into your Portia [**dashboard ↗**](https://app.portialabs.ai)
2. Navigate to the **Manage Org** tab.
3. Enter the custom URL in 'Org Settings'. Ensure that the URL begins with either `https://` or `http://`.

## Bringing the concepts together[​](#bringing-the-concepts-together "Direct link to Bringing the concepts together")

Now let's bring this to life by reproducing the experience that you can see on the website's playground ([**↗**](https://www.portialabs.ai)). We want to be able to handle a prompt like `Find the github repository of Mastodon and give it a star for me`, so let's take a look at the code below.

**Portia API key required**

We're assuming you already have a Portia API key from the dashboard and set it in your environment variables. If not please refer to the previous section and do that first ([**Set up your account ↗**](/setup-account)).

main.py

```
from dotenv import load_dotenv  
from portia import (  
    ActionClarification,  
    InputClarification,  
    MultipleChoiceClarification,  
    PlanRunState,  
    Portia,  
    PortiaToolRegistry,  
    default_config,  
)  
  
load_dotenv()  
  
# Instantiate a Portia instance. Load it with the default config and with Portia cloud tools above  
portia = Portia(tools=PortiaToolRegistry(default_config()))  
  
# Run Portia to star the repository  
plan_run = portia.run('Find the github repository of PortiaAI and give it a star for me')  
  
while plan_run.state == PlanRunState.NEED_CLARIFICATION:  
    # If clarifications are needed, resolve them before resuming the plan run  
    for clarification in plan_run.get_outstanding_clarifications():  
        # Usual handling of Input and Multiple Choice clarifications  
        if isinstance(clarification, (InputClarification, MultipleChoiceClarification)):  
            print(f"{clarification.user_guidance}")  
            user_input = input("Please enter a value:\n"   
                            + (("\n".join(clarification.options) + "\n") if "options" in clarification else ""))  
            plan_run = portia.resolve_clarification(clarification, user_input, plan_run)  
          
        # Handling of Action clarifications  
        if isinstance(clarification, ActionClarification):  
            print(f"{clarification.user_guidance} -- Please click on the link below to proceed.")  
            print(clarification.action_url)  
            plan_run = portia.wait_for_ready(plan_run)  
  
    # Once clarifications are resolved, resume the plan run  
    plan_run = portia.resume(plan_run)  
  
# Serialise into JSON and print the output  
print(f"{plan_run.model_dump_json(indent=2)}")
```

Pay attention to the following points:

* We're importing all of Portia's cloud tool library using the `PortiaToolRegistry` import. Portia will (rightly!) identify that executing on this query necessitates both the `SearchGitHubReposTool` and the `StarGitHubRepoTool` in particular. Like all Portia cloud tools, our Github tools are built with plug and play authentication support. Before any steps are executed, Portia will raise an `Action Clarification` with a Github OAuth link as the action URL.
* We're now introducing the `portia.wait_for_ready()` method to handle clarifications of type `ActionClarification`. This method should be used when the resolution to a clarification relies on a third party system and your `Portia` instance needs to listen for a change in its state. In our example, Portia's OAuth server listens for the authentication result and resolves the concerned clarification, allowing the plan run to resume again.

Your plan run will pause and you should see the link in the logs like so
...

```
OAuth required -- Please click on the link below to proceed.  
https://github.com/login/oauth/authorize/?redirect_uri=https%3A%2F%2Fapi.portialabs.ai%2Fapi%2Fv0%2Foauth%2Fgithub%2F&client_id=Ov23liXuuhY9MOePgG8Q&scope=public_repo+starring&state=APP_NAME%3Dgithub%253A%253Agithub%26PLAN_RUN_ID%3Daa6019e1-0bde-4d76-935d-b1a64707c64e%26ORG_ID%3Dbfc2c945-4c8a-4a02-847a-1672942e8fc9%26CLARIFICATION_ID%3D9e6b8842-dc39-40be-a298-900383dd5e9e%26SCOPES%3Dpublic_repo%2Bstarring&response_type=code
```

In your logs you should be able to see the tools, as well as a plan and final plan run state similar to the output below. Note again how the planner weaved tools from both the cloud and the example registry.

* Generated plan
* Plan run in final state

plan-71fbe578-0c3f-4266-b5d7-933e8bb10ef2.json

```
{  
    "id": "plan-71fbe578-0c3f-4266-b5d7-933e8bb10ef2",  
    "plan_context": {  
        "query": "Find the github repository of PortiaAI and give it a star for me",  
        "tool_ids": [  
        "portia::github::search_repos",  
        "portia::github::star_repo",  
        "portia::slack::send_message",  
        "portia::zendesk::list_groups_for_user",  
        ...  
        ]  
    },  
    "steps": [  
        {  
            "task": "Search for the GitHub repository of PortiaAI",  
            "inputs": [],  
            "tool_id": "portia:github::search_repos",  
            "output": "$portiaai_repository"  
        },  
        {  
        "task": "Star the GitHub repository of PortiaAI",  
        "inputs": [  
            {  
                "name": "$portiaai_repository",  
                "description": "The GitHub repository of PortiaAI"  
            }  
        ],  
        "tool_id": "portia:github::star_repo",  
        "output": "$star_result"  
        }  
    ]  
}
```

prun-36945fae-1dcc-4b05-9bc4-4b862748e031.json

```
{  
    "id": "prun-36945fae-1dcc-4b05-9bc4-4b862748e031",  
    "plan_id": "plan-71fbe578-0c3f-4266-b5d7-933e8bb10ef2",  
    "current_step_index": 1,  
    "state": "COMPLETE",  
    "outputs": {  
        "clarifications": [  
            {  
                "uuid": "clar-f873b9be-10ee-4184-a717-3a7559416499",  
                "category": “Multiple Choice”,  
                "response": “portiaAI/portia-sdk-python",  
                "step": 2,   
                "user_guidance": "Please select a repository.",   
                "handled": true,  
                "argument": "$portiaai_repository",  
                "options": "[\"portiaAI/portia-sdk-python\", \"portiaAI/docs\", \"portiaAI/portia-agent-examples\"]",  
            }  
        ],  
        "step_outputs": {  
        "$portiaai_repository": {  
            "value": "[\"portiaAI/portia-sdk-python\", \"portiaAI/docs\", \"portiaAI/portia-agent-examples\"]",  
            "summary": null  
        },  
        "$star_result": {  
            "value": "Successfully starred the repository 'portiaAI/portia-sdk-python'.",  
            "summary": null  
        }  
        },  
        "final_output": {  
        "value": "Successfully starred the repository 'portiaAI/portia-sdk-python'.",  
        "summary": null  
        }  
    }  
}
```

info

Now that you're familiar with running Portia tools, why not try your hand at the intro example in our [**examples repo (↗)**](https://github.com/portiaAI/portia-agent-examples/blob/main/get_started_google_tools/README.md). In the example ee use the Google Calendar tools to schedule a meeting and handle the authentication process to execute those tool calls.

Last updated on **Sep 9, 2025** by **robbie-portia**