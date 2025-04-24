import dotenv

from portia import Config, LogLevel, Portia
from portia.cli import CLIExecutionHooks
from portia.tool_registry import McpToolRegistry

dotenv.load_dotenv()

tools = McpToolRegistry.from_sse_connection(
    server_name="my_oauth_mcp",
    url="https://mcp-github-oauth.sam-f86.workers.dev",
    use_oauth=True,
)

portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=tools,
    execution_hooks=CLIExecutionHooks(),
)

# Simple Example
plan_run = portia.run(
    "Get my user info",
)
