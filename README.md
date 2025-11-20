<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Logo_Portia_Line_White.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/Logo_Portia_Line_Black.png">
    <img alt="Portia AI logo" src="https://raw.githubusercontent.com/portiaAI/portia-sdk-python/main/assets/Logo_Portia_Line_Black.png"  width="50%">
  </picture>
</p>

# Portia SDK Python

> **⚠️ Important Notice: Repository Deprecation and Transition to Closed Source**
> 
> A commonly known secret in the open source industry is that most companies have an uncomfortable relationship with their open source product at some point in their lifetime. As the Portia product has evolved, it has become increasingly clear that the product that we've built works best for real scaled production deployments. For this reason, we've made the difficult decision to no longer be open source.
> 
> To our open source users, firstly, thank you for your contributions and investment in making Portia what it is today. We have yanked all versions on pip but not deleted them – our cut off date for doing so will be the end of November. Likewise, the repository will change to private at the end of November and future evolutions of Portia will be closed source.
> 
> If you are using the cloud based product, please get in touch at hello@portialabs.ai to discuss your specific usecase.
> 
> Thank you again,
> 
> The Portia Team

## Key features

**Iterate on agents’ reasoning and intervene in their execution**</br>
🧠 Create your multi-agent [`Plan`](https://docs.portialabs.ai/generate-plan) conversationally, or build them with our [`PlanBuilder`](https://docs.portialabs.ai/generate-plan#build-a-plan-manually).</br>
📝 Enrich a [`PlanRunState`](https://docs.portialabs.ai/run-plan) during execution to track progress.</br>
🚧 Define inputs and output structures for enhanced predictability.</br>
✋🏼 Add deterministic tasks through an [`ExecutionHook`](https://docs.portialabs.ai/execution-hooks). Use a [`clarification`](https://docs.portialabs.ai/understand-clarifications) for human:agent interactions.</br>

**Extensive tool support including MCP support**</br>
🔧 Connect [tool registries](https://docs.portialabs.ai/extend-run-tools) from any MCP server, local tools or another AI tool provider (e.g. ACI.dev).</br>
🫆 Leverage Portia cloud's prebuilt [1000+ cloud and MCP tools](https://docs.portialabs.ai/cloud-tool-registry) with out-of-the-box authentication.</br>
🌐 Navigate the web and cope with captchas and logins using our [open source browser tool](https://docs.portialabs.ai/browser-tools).</br>

**Authentication for API and web agents**</br>
🔑 Handle user credentials seamlessly, for both API tools and browser sessions, with our `clarification` interface.</br>

**Production ready**</br>
👤 Attribute multi-agent runs and auth at an [`EndUser`](https://docs.portialabs.ai/manage-end-users) level.</br>
💾 Large inputs and outputs are automatically stored / retrieved in [Agent memory](https://docs.portialabs.ai/agent-memory) at runtime.</br>
🔗 Connect [any LLM](https://docs.portialabs.ai/manage-config#configure-llm-options) including local ones, and use your own Redis server for [caching](https://docs.portialabs.ai/manage-config#manage-caching).</br>


## Demo
To clickthrough at your own pace, please follow this [link](https://snappify.com/view/3d721d6c-c5ff-4e84-b770-83e93bd1a8f1)</br>
![Feature run-through](https://github.com/user-attachments/assets/1cd66940-ee78-42a6-beb4-7533835de7e9)

## Quickstart

### Installation in 3 quick steps

Ensure you have python 3.11 or higher installed using `python --version`. If you need to update your python version please visit their [docs](https://www.python.org/downloads/). Note that the example below uses OpenAI but we support other models as well. For instructions on linking other models, refer to our [docs](https://docs.portialabs.ai/manage-config).</br>

**Step 1:** Install the Portia Python SDK
```bash
pip install portia-sdk-python 
```
> **Note:** All versions have been yanked from pip. Existing installations may continue to work, but no new versions will be published. The repository will become private at the end of November.

**Step 2:** Ensure you have an LLM API key set up
```bash
export OPENAI_API_KEY='your-api-key-here'
```
**Step 3:** Validate your installation by submitting a simple maths prompt from the command line
```
portia-cli run "add 1 + 2"
```

**All set? Now let's explore some basic usage of the product 🚀**

### E2E example
You will need a Portia API key* for this one because we use one of our cloud tools to schedule a calendar event and send an email. 
<br>**🙏🏼 *We have a free tier so you do not need to share payment details to get started 🙏🏼.**<br>
Head over to <a href="https://app.portialabs.ai" target="_blank">**app.portialabs.ai (↗)**</a> and get your Portia API key. You will then need to set it as the env variable `PORTIA_API_KEY`.<br/>

The example below introduces **some** of the config options available with Portia AI (check out our <a href="https://docs.portialabs.ai/manage-config" target="_blank">**docs (↗)**</a> for more):
- The `storage_class` is set using the `StorageClass.CLOUD` ENUM. So long as your `PORTIA_API_KEY` is set, runs and tool calls will be logged and appear automatically in your Portia dashboard at <a href="https://app.portialabs.ai" target="_blank">**app.portialabs.ai (↗)**</a>.
- The `default_log_level` is set using the `LogLevel.DEBUG` ENUM to `DEBUG` so you can get some insight into the sausage factory in your terminal, including plan generation, run states, tool calls and outputs at every step 😅
  - To enable ultra-verbose tracing of LLM calls across agents and tools, set `default_log_level=LogLevel.TRACE` (or the string "TRACE"). TRACE includes all DEBUG logs plus additional "LLM call" entries showing the model and high-level purpose (planning, introspection, summarization, parsing/verification, tool-calling).
- The `llm_provider` and `xxx_api_key` (varies depending on model provider chosen) are used to choose the specific LLM provider. In the example below we're using GPT 4o, but you can use Anthropic, Gemini, Grok and others!

Finally we also introduce the concept of a `tool_registry`, which is a flexible grouping of tools.

```python
from dotenv import load_dotenv
from portia import Config, Portia, DefaultToolRegistry
from portia.cli import CLIExecutionHooks

load_dotenv(override=True)

recipient_email = input("Please enter the email address of the person you want to schedule a meeting with:\n")
task = f"""
Please help me accomplish the following tasks:
- Get my availability from Google Calendar tomorrow between 8:00 and 8:30
- If I am available, schedule a 30 minute meeting with {recipient_email} at a time that works for me with the title "Portia AI Demo" and a description of the meeting as "Test demo".
"""

config = Config.from_default()
portia = Portia(
   config=config,
   tools=DefaultToolRegistry(config=config),
   execution_hooks=CLIExecutionHooks(),
)

plan = portia.run(task)
```

### Advanced examples on YouTube
Here is an example where we build a customer refund agent using Stripe's MCP server. It leverages execution hooks and clarifications to confirm human approval before moving money.</br>
[![Customer refund agent with Stripe MCP](assets/stripemcp.jpg)](https://youtu.be/DB-FDEM_7_Y?si=IqVq14eskvLIKmvv)

Here is another example where we use our open browser tool. It uses clarifications when it encounters a login page to allow a human to enter their credentials directly into the session and allow it to progress.</br>
[![Manage Linkedin connections](assets/linkedinbrowsertool.jpg)](https://youtu.be/hSq8Ww-hagg?si=8oQaXcTcAyrzEQty)

## Learn more
- Head over to our docs at <a href="https://docs.portialabs.ai" target="_blank">**docs.portialabs.ai (↗)**</a>.
- For questions about the cloud-based product, please contact us at hello@portialabs.ai.

# ❤️ Contributors
A heartfelt thank you to our growing list of contributors!
<!-- readme: contributors,mounir-portia/-,robbie-portia/-,emmaportia/-,sam-portia/-,Nathanjp91/-,TomSPortia/-,OmarEl-Mohandes/-,portiaAI-bot/- -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/fanlinwang">
                    <img src="https://avatars.githubusercontent.com/u/25503130?v=4" width="100;" alt="fanlinwang"/>
                    <br />
                    <sub><b>fanlinwang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/hannah-portia">
                    <img src="https://avatars.githubusercontent.com/u/187275200?v=4" width="100;" alt="hannah-portia"/>
                    <br />
                    <sub><b>hannah-portia</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/asquare14">
                    <img src="https://avatars.githubusercontent.com/u/22938709?v=4" width="100;" alt="asquare14"/>
                    <br />
                    <sub><b>Atibhi Agrawal</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/judy2k">
                    <img src="https://avatars.githubusercontent.com/u/359772?v=4" width="100;" alt="judy2k"/>
                    <br />
                    <sub><b>Mark Smith</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/md-abid-hussain">
                    <img src="https://avatars.githubusercontent.com/u/101964499?v=4" width="100;" alt="md-abid-hussain"/>
                    <br />
                    <sub><b>Md Abid Hussain</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/saitama142">
                    <img src="https://avatars.githubusercontent.com/u/53933154?v=4" width="100;" alt="saitama142"/>
                    <br />
                    <sub><b>0u55</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/amorriscode">
                    <img src="https://avatars.githubusercontent.com/u/16005567?v=4" width="100;" alt="amorriscode"/>
                    <br />
                    <sub><b>Anthony Morris</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/gaurava05">
                    <img src="https://avatars.githubusercontent.com/u/45741026?v=4" width="100;" alt="gaurava05"/>
                    <br />
                    <sub><b>Gaurav Agarwal</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jingkaihe">
                    <img src="https://avatars.githubusercontent.com/u/1335938?v=4" width="100;" alt="jingkaihe"/>
                    <br />
                    <sub><b>Jingkai He</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ksingh-08">
                    <img src="https://avatars.githubusercontent.com/u/148441103?v=4" width="100;" alt="ksingh-08"/>
                    <br />
                    <sub><b>Karan Singh</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/account">
                    <img src="https://avatars.githubusercontent.com/u/186828738?v=4" width="100;" alt="account"/>
                    <br />
                    <sub><b>Swagatika Beura</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/account">
                    <img src="https://avatars.githubusercontent.com/u/186828738?v=4" width="100;" alt="account"/>
                    <br />
                    <sub><b>Swagatika Beura</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors,mounir/-,robbie-portia/-,emmaportia/-,sam-portia/-,Nathanjp91/-,tomSportia/-,OmarEl-Mohandes/- -end -->
