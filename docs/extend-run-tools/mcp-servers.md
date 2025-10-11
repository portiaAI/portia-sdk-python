* [Extend and run tools](/extend-run-tools)
* Integrating an MCP server with the SDK

# Integrating an MCP server with the SDK

The Model Context Protocol (MCP) makes it very easy to integrate third-party tools into your Portia AI project.
To find out more you can visit the [official MCP docs ↗](https://modelcontextprotocol.io/).

Remote MCP servers

This section covers integration an MCP server with the **SDK**. The easiest way to connect a remote MCP server with managed authentication is through the [Cloud tool registry](https://app.portialabs.ai/dashboard/tool-registry) (see documentation [here](/cloud-tool-registry)). When using the cloud tool registry, authentication is handled for you, but if you're connecting directly into the SDK then you'll need to handle authentication yourself.

We provide developers with the ability to set up their own MCP server connections (local or remote) directly into our SDK. When integrating an SDK server directly with the SDK, we offer the three methods currently available for interacting with MCP servers:

* **STDIO** (Standard input/output): The server runs as a subprocess of your main python process. Below we interact with that process via an npx command and a docker command provided with the correct arguments.
* **Streamable HTTP**: Communication is over HTTP, you can run the server locally or deploy a server remotely. Per below, you just need to specify the current server name and URL.
* **SSE** (Server-Sent Events): A legacy method of communication over HTTP, since replaced by Streamable HTTP.

To find out more about these options, see the official MCP docs ([**↗**](https://modelcontextprotocol.io/docs/concepts/transports)).

The `server_name` argument is used by Portia to identify where tools have come from, you can set this to whatever makes most sense for the MCP server you are integrating. If you re-use old `Plan` objects later on, make sure to use the same `server_name` with the MCP server.

* STDIO
* Streamable HTTP
* SSE

mcp\_stdio\_example.py

```
import os  
  
from portia import (  
    DefaultToolRegistry,  
    Portia,  
    McpToolRegistry,  
    Config,  
)  
  
config = Config.from_default()  
  
tool_registry = (  
    # Integrates the Stripe MCP server from   
    # https://github.com/stripe/agent-toolkit/tree/main/modelcontextprotocol  
    McpToolRegistry.from_stdio_connection(  
        server_name="stripe",  
        command="npx",  
        args=[  
            "-y",  
            "@stripe/mcp",  
            "--tools=all",  
            f"--api-key={os.getenv('STRIPE_API_KEY')}",  
        ],  
    )  
    # Integrates Github MCP server using docker  
    + McpToolRegistry.from_stdio_connection(  
        server_name="github",  
        command="docker",  
        args=[  
            "run",  
            "-i",  
            "--rm",  
            "-e",  
            "GITHUB_PERSONAL_ACCESS_TOKEN",  
            "ghcr.io/github/github-mcp-server"  
        ],  
        env={  
            "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR TOKEN>"  
        }  
    )  
    + DefaultToolRegistry(config)  
)  
  
portia = Portia(config=config, tools=tool_registry)
```

mcp\_streamable\_http\_example.py

```
from portia import (  
    DefaultToolRegistry,  
    Portia,  
    McpToolRegistry,  
    Config,  
)  
  
config = Config.from_default()  
  
tool_registry = (  
    # Assumes server is running on port 8000  
    McpToolRegistry.from_streamable_http_connection(  
        server_name="mcp_streamable_http_example_server",  
        url="http://mcp.example.com/http",  
    )  
    + DefaultToolRegistry(config)  
)  
  
portia = Portia(config=config, tools=tool_registry)
```

mcp\_sse\_example.py

```
from portia import (  
    DefaultToolRegistry,  
    Portia,  
    McpToolRegistry,  
    Config,  
)  
  
config = Config.from_default()  
  
tool_registry = (  
    # Assumes server is running on port 8000  
    McpToolRegistry.from_sse_connection(  
        server_name="mcp_sse_example_server",  
        url="http://localhost:8000",  
    )  
    + DefaultToolRegistry(config)  
)  
  
portia = Portia(config=config, tools=tool_registry)
```

Pre-requisites

To run the stdio example, make sure `npx` and `docker` are available in your environment. Many MCP servers are currently provided to run in this way, usually either run with the `npx`, `docker` or `uvx` command.

When you provide a `McpToolRegistry`, Portia will pull in the tool definitions from the MCP server, making them available to the Planner and Execution Agents during a plan run. To see an example of this implementation, head over to our agent-examples repo where we built an agent to manage customer refunds ([**↗**](https://github.com/portiaAI/portia-agent-examples/tree/main/refund-agent-mcp)).

There are many open source MCP servers already available: check out the list of servers on the official MCP github repository ([**↗**](https://github.com/modelcontextprotocol/servers)).

Last updated on **Sep 9, 2025** by **robbie-portia**