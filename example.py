import dotenv

from portia import Config, LogLevel, Portia
from portia.cli import CLIExecutionHooks
from portia.tool_registry import McpToolRegistry

dotenv.load_dotenv()

tools = McpToolRegistry.from_sse_connection(
    server_name="linear-test",
    url="https://mcp.linear.app",
    use_oauth=True,
)

portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=tools,
    execution_hooks=CLIExecutionHooks(),
)

# Simple Example
plan_run = portia.run(
    "List my Linear issues",
)