* [Extend and run tools](/extend-run-tools)
* Integrating tools

On this page

# Integrating tools

Learn how to integrate tools that your agent can use to answer a user query.

TL;DR

* You can specify the tools that agents can use to answer a user query by using the `tools` argument in your `Portia` instance. If you don't specify this, the `Portia` instance will use a default set of tools.
* Tool registries are useful to group frequently used tools together. They are represented by the `ToolRegistry` class ([**SDK reference ↗**](/SDK/portia/tool_registry)).

## Overview of tool integration[​](#overview-of-tool-integration "Direct link to Overview of tool integration")

As part of defining your `Portia` instance for a query, you can specify the tools that the LLM can use to answer the query. This is done by specifying the `tools` argument in the `Portia` instance definition.

```
from portia import (  
  default_config,   
  Portia,  
)  
from portia.open_source_tools.calculator_tool import CalculatorTool  
from portia.open_source_tools.search_tool import SearchTool  
from portia.open_source_tools.weather import WeatherTool  
  
# Instantiate a Portia instance. Load it with the default config and with the example tools.  
portia = Portia(tools=[CalculatorTool(), SearchTool(), WeatherTool()])
```

If you don't specify the `tools` argument, your `Portia` instance will use a default set of tools.

Default tools

The default tool set comprises:

* The [**open source tool set**](/portia-tools/open-source/), with the Search tool and Weather tool only included if you have the corresponding Tavily / OpenWeatherMap API keys specified.
* If you have an API key for Portia Cloud, the tools from your cloud tool registry will be included. This includes the ability to integrate any remote MCP server, as well as a suite of pre-created integrations you can use straight off the bat.
  Further information on this tool registry, including how it can be configured, can be found on the [**Remote MCP and cloud tools page ↗**](/cloud-tool-registry).

## Tool registries[​](#tool-registries "Direct link to Tool registries")

A tool registry is a collection of tools and is represented by the `ToolRegistry` class ([**SDK reference ↗**](/run-portia-tools)). Tool registries are useful to group frequently used tools together, e.g. you could imagine having a tool registry by function in your organisation. Portia's default tool registry can be accessed by calling `DefaultToolRegistry(config=default_config())`.

```
from dotenv import load_dotenv  
from portia import (  
    DefaultToolRegistry,  
    Portia,  
    default_config,  
)  
from portia.open_source_tools.calculator_tool import CalculatorTool  
from portia.open_source_tools.search_tool import SearchTool  
from portia.open_source_tools.weather import WeatherTool  
  
load_dotenv()  
  
# Instantiate a Portia instance. Load it with the example tools and Portia's tools.  
portia = Portia(tools=DefaultToolRegistry(default_config()))
```

### Customizing tool descriptions[​](#customizing-tool-descriptions "Direct link to Customizing tool descriptions")

It's often the case that you want to provide custom instructions to Portia agents about how to use a tool, for example, because the author of the MCP tool has missed some context that's important for your usecase, or because you want to personalize the tool in some way. We offer an easy way to edit tool descriptions to do this using the `ToolRegistry.with_tool_description` function.

Consider the below example that personalizes the Linear MCP server with the default team ID:

customize\_tool\_descriptions.py

```
from portia import Config, Portia, PortiaToolRegistry  
from portia.cli import CLIExecutionHooks  
  
my_config = Config.from_default()  
  
portia = Portia(  
    config=my_config,  
    tools=PortiaToolRegistry(my_config).with_tool_description(  
        "portia:mcp:custom:mcp.linear.app:create_issue",  
        "If a teamID is not provided, use teamID 123."),  
    execution_hooks=CLIExecutionHooks(),  
)
```

This customization can be used across any tool registry in Portia.

## Available tools[​](#available-tools "Direct link to Available tools")

When setting up your tool registry, there are four sources of tools you can use: our open-source tools, our Portia cloud tools, your own MCP tool registry and custom code tools.

### Open source tools[​](#open-source-tools "Direct link to Open source tools")

Portia provides an open source tool registry that contains a selection of general-purpose utility tools. For example, it includes a Tavily tool for web search, an OpenWeatherMap tool for determining weather and a PDF reader tool, among many others.
The open source tool registry can be used as follows, though for some of the tools you will need to retrieve an API key first:

```
from portia import open_source_tool_registry, Portia  
  
portia = Portia(tools=open_source_tool_registry)
```

For more details, check out our [open-source tool documentation ↗](/portia-tools/open-source/).

### Portia cloud registry[​](#portia-cloud-registry "Direct link to Portia cloud registry")

Portia cloud provides an extensive tool registry to speed up your agent development, with authentication handled seamlessly by Portia for you.
You can select any MCP server with an official remote server implementation from our Tool registry dashboard and connect it to your account. We are rapidly growing our library as providers bring out new remote MCP servers. If you'd like to add a missing or proprietary remote MCP server to your Portia cloud registry and rely on Portia to handle authentication for you, you can do that from the dashboard as well.
Finally Portia cloud also includes some in-house-built tools that don't have an official MCP server implementation e.g. Google and Microsoft productivity tools.

Your Portia tool registry is available through the `PortiaToolRegistry` class ([**SDK reference ↗**](/run-portia-tools)). This gives access to all the tools you have enabled in your registry:

```
from portia import Portia, PortiaToolRegistry, Config  
  
config = Config.from_default()  
portia = Portia(tools=PortiaToolRegistry(config))
```

More details can be found on our [Cloud tool registry ↗](/cloud-tool-registry) page, including how to enable / disable tools within the registry and how to connect in your own remote MCP server.

### Integrate your own MCP servers [SDK-only option][​](#integrate-your-own-mcp-servers-sdk-only-option "Direct link to Integrate your own MCP servers [SDK-only option]")

You can easily add any local or remote MCP servers directly into a Portia agent through our `McpToolRegistry` class.
The key difference between integrating an MCP server this way and through the Portia cloud registry is that authentication needs to be handled manually when integrating directly into the Portia instance.
The MCP server can be added to your Portia instance as follows, with more details available on our [integrating MCP servers ↗](/mcp-servers) page.

```
from portia import Portia, McpToolRegistry  
  
tool_registry = (  
    # Assumes server is running on port 8000  
    McpToolRegistry.from_sse_connection(  
        server_name="mcp_sse_example_server",  
        url="http://localhost:8000",  
    )  
)  
portia = Portia(tools=tool_registry)
```

### Custom tools[​](#custom-tools "Direct link to Custom tools")

As outlined in the [Introduction to tools ↗](/mcp-servers), it is easy to define your own tools in python code with Portia. In ([**Adding custom tools ↗**](/add-custom-tools)), we'll walk through how to do this in more detail by creating our own tool registries with custom tools.

## Filtering tool registries[​](#filtering-tool-registries "Direct link to Filtering tool registries")

You can create new tool registries from existing ones by filtering tools to your desired subset. For example, you might want to prevent one of your agents from accessing emails in Gmail. This can be done by setting up a filter to exclude the Gmail tools from the registry:

```
from dotenv import load_dotenv  
from portia import (  
    Portia,  
    PortiaToolRegistry,  
    Tool,  
    default_config,  
)  
  
load_dotenv()  
  
def exclude_gmail_filter(tool: Tool) -> bool:  
    return not tool.id.startswith("portia:google:gmail:")  
  
registry = PortiaToolRegistry(config=default_config()).filter_tools(exclude_gmail_filter)  
portia = Portia(tools=registry)
```

Last updated on **Sep 9, 2025** by **robbie-portia**