"""Simple Example."""

import os
from portia import (
    Config,
    LogLevel,
    McpToolRegistry,
    PlanRunState,
    Portia,
    StdioMcpClientConfig,
    example_tool_registry,
    execution_context,
)

portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=McpToolRegistry(
        StdioMcpClientConfig(
            server_name="tavily",
            command="npx",
            args=["-y", "tavily-mcp@0.1.4"],
            env={
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY") or "",
            },
        ),
    ),
)


# Simple Example
plan_run = portia.run(
    "Whats the latest news on Trumps tariffs from the current week",
)
