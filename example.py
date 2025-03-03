"""Simple Example."""

from portia import (
    Config,
    LogLevel,
    PlanRunState,
    Portia,
    example_tool_registry,
    execution_context,
)
from portia.cli import CLIExecutionHooks

portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry,
)


# Simple Example
plan_run = portia.run(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)

# We can also provide additional execution context to the process
with execution_context(end_user_id="123", additional_data={"email_address": "hello@portialabs.ai"}):
    plan = portia.run(
        "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
    )

# When we hit a clarification we can ask our end user for clarification then resume the process
with execution_context(end_user_id="123", additional_data={"email_address": "hello@portialabs.ai"}):
    plan_run = portia.run(
        "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
    )

# Fetch run
plan_run = portia.storage.get_plan_run(plan_run.id)
# Update clarifications
if plan_run.state == PlanRunState.NEED_CLARIFICATION:
    for c in plan_run.get_outstanding_clarifications():
        # Here you prompt the user for the response to the clarification
        # via whatever mechanism makes sense for your use-case.
        new_value = "Answer"
        plan_run = portia.resolve_clarification(
            plan_run=plan_run,
            clarification=c,
            response=new_value,
        )

# Execute again with the same execution context
with execution_context(context=plan_run.execution_context):
    portia.resume(plan_run)

# You can also pass in a clarification handler to manage clarifications
portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry,
    execution_hooks=CLIExecutionHooks(),
)
plan_run = portia.run(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)
