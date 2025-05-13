"""Simple Example."""

from dotenv import load_dotenv

from portia import (
    Config,
    LogLevel,
    Portia,
    open_source_tool_registry,
)
from portia.cli_clarification_handler import CLIClarificationHandler
from portia.config import default_config
from portia.execution_hooks import (
    ExecutionHooks,
    clarify_on_tool_call,
    log_step_outputs,
)
from portia.tool_registry import PortiaToolRegistry

load_dotenv()

tool_registry = registry = (
    # Exclude gmail tools and leave outlook tools
    PortiaToolRegistry(default_config()).filter_tools(
        lambda tool: not tool.id.startswith("portia:google:gmail:")
    )
    + open_source_tool_registry
)
portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=tool_registry,
    execution_hooks=ExecutionHooks(
        clarification_handler=CLIClarificationHandler(),
        before_step_execution=lambda plan, plan_run, step: print(
            f"Before step: plan: {plan}, plan_run: {plan_run}, step: {step}"
        ),
        after_step_execution=log_step_outputs,
        before_first_step_execution=lambda plan, plan_run: print(
            f"Before first step, plan: {plan}, plan_run: {plan_run}"
        ),
        after_last_step_execution=lambda plan, plan_run, output: print(
            f"After last step, plan: {plan}, plan_run: {plan_run}, output: {output}"
        ),
        before_tool_call=clarify_on_tool_call("portia:microsoft:outlook:send_email"),
        after_tool_call=lambda tool, args, plan_run, step: print(
            f"After tool call {tool}, args: {args}, plan_run: {plan_run}, step: {step}"
        ),
    ),
)


# Simple Example
plan_run = portia.run(
    "Send an email to Robbie at robbie@portilabs.ai telling him how fantastic he is, "
    "then get the weather in London",
)
