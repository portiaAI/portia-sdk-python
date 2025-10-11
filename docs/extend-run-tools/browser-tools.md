* [Extend and run tools](/extend-run-tools)
* Using browser tools

On this page

# Using browser tools

Browser tools ([**SDK ↗**](/SDK/portia/open_source_tools/browser_tool)) can deploy an agent to browse the internet and retrieve data or enact actions on your behalf. Portia will use Browser tools when it recognises there is a web-based task to be performed. We use the [**Browser Use (↗)**](https://browser-use.com) library to offer a multi-modal web agent that will visually and textually analyse a website in order to navigate it and carry out a task.

Our browser tool can be used in two modes:

* **Remote mode**: Runs on a remote chromium instance using [**Browserbase (↗)**](https://www.browserbase.com/) as the underlying infrastructure. Browserbase offers infrastructure for headless browsers remotely. We spin up remote sessions for your end-users which persist through clarifications.
* **Local mode (DEFAULT)**: Runs on a chrome instance on your own computer. Requires Chrome to be started fresh by the agent to work.

The underlying library for navigating the page is provided by [**Browser Use (↗)**](https://browser-use.com). It uses a number of LLM calls to navigate the page and complete the action.

## Setting up the browser based tools[​](#setting-up-the-browser-based-tools "Direct link to Setting up the browser based tools")

* Browserbase setup
* Local setup

To use browserbase infrastructure, you need to install the required `tools-browser-browserbase` dependency group:

```
pip install "portia-sdk-python[tools-browser-browserbase]"  
# Alternatively, install our 'all' dependency group to get everything  
pip install "portia-sdk-python[all]"
```

You must also ensure that you have set the `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID` in your .env file (or equivalent). These can be obtained by creating an account on [**Browserbase (↗)**](https://www.browserbase.com). The current behaviour requires a paid version of Browserbase to use.

With local setup, the browser tool uses chrome on the machine it is running on. This means that it is not possible to support end-users but is a good way to test or to write agents for your own purposes. To use the browser tool in local mode, you need to install the required `tools-browser-local` dependency group:

```
pip install "portia-sdk-python[tools-browser-local]"  
# Alternatively, install our 'all' dependency group to get everything  
pip install "portia-sdk-python[all]"
```

You must then specify the `BrowserInfrastructureOption` when creating the tool, i.e:

```
from portia.open_source_tools.browser_tool import BrowserInfrastructureOption, BrowserTool  
browser_tool = BrowserTool(infrastructure_option=BrowserInfrastructureOption.LOCAL)
```

You can specify the executable for the Chrome instance, by setting `PORTIA_BROWSER_LOCAL_CHROME_EXEC='path/to/chrome/exec'`. If not specified, the default location on most operating systems will be used. For the agent to work, all other Chrome instances must be closed before the task starts.

## Using browser based tools in Portia[​](#using-browser-based-tools-in-portia "Direct link to Using browser based tools in Portia")

The `BrowserTool` is located in our open source tools folder [**SDK ↗**](/SDK/portia/open_source_tools/browser_tool.py). Additionally, there are 2 ways to use the tool:

* **`BrowserTool()`**: This is a general browser tool and it will be used when a URL is provided as part of the query.

BrowserTool example

```
from portia import Config, Portia  
from portia.open_source_tools.browser_tool import BrowserTool  
  
task = "Find my connections called 'Bob' on LinkedIn (https://www.linkedin.com)"  
  
# Needs BrowserBase API key and project_id  
portia = Portia(config=Config.from_default(),  
                tools=[BrowserTool()])
```

* **`BrowserToolForUrl(url)`**: To restrict the browser tool to a specific URL. This is particularly useful to ensure that the planner is restricted to the domains that you want it to be support.

BrowserToolForUrl example

```
from portia import Config, Portia  
from portia.open_source_tools.browser_tool import BrowserToolForUrl  
  
task = "Find my connections called 'Bob' on LinkedIn"  
  
# Needs BrowserBase API key and project_id  
portia = Portia(config=Config.from_default(),  
                tools=[BrowserToolForUrl("https://www.linkedin.com")])
```

### A simple E2E example[​](#a-simple-e2e-example "Direct link to A simple E2E example")

Full example

```
from dotenv import load_dotenv  
  
from portia import (  
    ActionClarification,  
    Config,  
    PlanRunState,  
    Portia,  
)  
from portia.open_source_tools.browser_tool import BrowserTool  
  
load_dotenv(override=True)  
  
task = "Get the top news headline from the BBC news website (https://www.bbc.co.uk/news)"  
  
portia = Portia(Config.from_default(), tools=[BrowserTool()])  
  
plan_run = portia.run(task)  
  
while plan_run.state == PlanRunState.NEED_CLARIFICATION:  
    # If clarifications are needed, resolve them before resuming the workflow  
    print("\nPlease resolve the following clarifications to continue")  
    for clarification in plan_run.get_outstanding_clarifications():  
        # Handling of Action clarifications  
        if isinstance(clarification, ActionClarification):  
            print(f"{clarification.user_guidance} -- Please click on the link below to proceed.")  
            print(clarification.action_url)  
            input("Press Enter to continue...")  
  
    # Once clarifications are resolved, resume the workflow  
    plan_run = portia.resume(plan_run)
```

## Authentication with browser based tools[​](#authentication-with-browser-based-tools "Direct link to Authentication with browser based tools")

Recap: Portia Authentication

Portia uses `Clarifications` to handle human-in-the-loop authentication (full explanation [**here ↗**](/run-portia-tools)). In our OAuth based tools, the user clicks on a link, authenticates and their token is used when the agents resumes.

In the browser tool case, whenever a browser tool encounters a page that requires authentication, it will raise a clarification request to the user, just like API-based Portia tools. The user will need to provide the necessary credentials or authentication information into the website to proceed. The cookies for that authentication are then used for the rest of the plan run.

![Browser authentication with clarifications](/assets/images/browser_auth-6d5d551dd0e9727ee9347ef4669411cf.png)

* Authentication with Browserbase
* Local Browser Authentication

In the case of Browserbase Authentication, the end-user will be provided with a URL starting with `browserbase.com/devtools-fullscreen/...`. When the end-user visits this page, they will see the authentication screen to enter their credentials (and any required 2FA or similar checks). This requires a paid version of Browserbase to work. In addition, Browserbase sessions have a timeout (default is 1hr with a max of 6hr) and the clarification must be handled by the user within this time.

Once the end-user has performed the authentication, they should then indicate to your application that they have completed the flow, and you should call `portia.resume(plan_run)` to resume the agent. Note if you are using the `CLIClarificationHandler`, this will not work in this way and you will need to override it to ensure this behaviour.

The authentication credentials will be saved against the end user and can be reused until they expire. When the credentials expire, a new clarification will be raised to reset the authentication. If you want to disable persistent authentication across agent runs, you should clear the `bb_context_id` attribute.

When running via a Local browser, i.e via your own computer, the clarification URL will be a regular URL that you can click on to authenticate.

## When to use API vs browser based tools[​](#when-to-use-api-vs-browser-based-tools "Direct link to When to use API vs browser based tools")

Browser based tools are very flexible in terms of what they do, however they do not have the same tight permissioning as OAuth tools and require more LLM calls so we recommend balancing between the two and using browser tools only when APIs are not available.

## Known issues and caveats[​](#known-issues-and-caveats "Direct link to Known issues and caveats")

### Popups and authentication[​](#popups-and-authentication "Direct link to Popups and authentication")

When using Browserbase as the underlying browser infrastructure, if authentication requires a popup, it will not show to the user and they will not be able to log-in. We are investigating solutions for this at the moment.

### Local chrome failing to connect[​](#local-chrome-failing-to-connect "Direct link to Local chrome failing to connect")

If you see an issue whereby Chrome opens, but then immediately closes and restarts, the issue is likely because it can't find the user data directory and the debug server is not starting. You can fix this by specifying the env variable `PORTIA_BROWSER_LOCAL_EXTRA_CHROMIUM_ARGS="--user-data-dir='path/to/dir'"` and there's more information about this on the [**browser-use issue link ↗**](https://github.com/browser-use/browser-use/issues/291#issuecomment-2792636861).

### LLM moderation[​](#llm-moderation "Direct link to LLM moderation")

We have occasionally observed that LLMs might get moderated on tasks that look like authentication requests to websites. These issues are typically transient but you may want to adjust the task or plan to avoid direct requests for the agent to login to a website.

Last updated on **Sep 9, 2025** by **github-actions[bot]**