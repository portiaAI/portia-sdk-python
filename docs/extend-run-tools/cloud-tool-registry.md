* [Extend and run tools](/extend-run-tools)
* Remote MCP and cloud tools

On this page

# Remote MCP and cloud tools

When your agents are connected to Portia Cloud, they gain access to an extensive tool registry with powerful integrations. The registry includes by default popular services like Gmail, Google Calendar, Slack, GitHub, Zendesk, and is extensible to many more by integrating remote MCP servers. You can check out and configure the integrations you want access to in [the dashboard (↗)](https://app.portialabs.ai/dashboard/tool-registry). This will update update the tools available to your `DefaultToolRegistry` (see [here](/integrating-tools#tool-registries) if you need a recap on how tool registries work).

Authentication for these tools is handled seamlessly by [Portia's authentication system ↗](/run-portia-tools). This means all tools are available using just the Portia API key and you don't have to worry about implementing OAuth flows or handling tokens and API keys yourself!

![Tool registry](/img/tool_registry.png)

A snippet of our tool registry

The registry contains applications, which are a collection of tools.
It is fully configurable, allowing you to turn applications on and off so you can control which tools your agents have access to.
The applications in the registry are a combination of remote MCP servers from official providers and tools developed by Portia.

## Remote MCP Servers[​](#remote-mcp-servers "Direct link to Remote MCP Servers")

The Model Context Protocol (MCP) makes it very easy to integrate third-party tools with Portia AI.
To find out more about MCP you can visit the official MCP docs ([**↗**](https://modelcontextprotocol.io/)).

We support remote MCP execution within our tool registry and, where possible, our integrations use remote MCP servers from official providers, with communication over a streamable HTTP connection.
This allows our tool registry to grow rapidly as providers bring out new remote MCP servers.
We support authentication natively for all of these servers and are in the process of adding many other features to make working with them easier.

You can extend your Portia cloud tool registry by configuring your own remote MCP server. This allows you to seamlessly integrate tools from any provider with a remote MCP server while Portia handles the authentication for you.

[](/img/register_mcp_server.mp4)

Connect your own MCP server into our cloud tool registry

Enabling authenticated remote MCP servers

It is worth noting that, when enabling MCP-based applications which use OAuth or API key authentication, you will need to authenticate with the server. This is required because MCP requires authentication in order to view available tools. The authentication credentials provided here are only used for listing tools from the server and are separate to those that the tool is executed with. We store all authentication credentials using [production-grade encryption](/security).

### Customizing MCP and other cloud based tools[​](#customizing-mcp-and-other-cloud-based-tools "Direct link to Customizing MCP and other cloud based tools")

We offer an easy way to customize our cloud based tools, or remote MCP server tool descriptions using the `ToolRegistry.with_tool_description` function. You can read more about this [here](/integrating-tools).

## Other Portia Cloud Tools[​](#other-portia-cloud-tools "Direct link to Other Portia Cloud Tools")

Where there is no official remote MCP server for a provider, we have a collection of tools developed by Portia.
This allows you to integrate easily with providers that are yet to release a remote MCP server.
Authentication for the tools is handled fully by the Portia platform and you can use these tools in exactly the same way as you can use tools coming from remote MCP servers.

## Enabling and Disabling Tools[​](#enabling-and-disabling-tools "Direct link to Enabling and Disabling Tools")

When you enable an application, all tools in this application become available to your agent. Applications can be easily enabled and disabled in the UI by:

1. Clicking on the 'Enable' / 'Disable' button when you hover over the application.
2. Configuring access if required - this is only required for remote MCP servers
3. Once this is done, the tool is configured and you'll be able to view the available tools under the application in the dashboard.

[](/img/tool_hover.mp4)

Quickly enable and disable tools hovering over them

It is important to choose your enabled tools carefully to avoid tool clashes. For example, if you wish to enable Microsoft Outlook, you should disable Gmail so that the agent knows which email provider to choose when you give it prompts like 'send an email'.

Last updated on **Sep 9, 2025** by **github-actions[bot]**