"""Simple Example."""

from dotenv import load_dotenv

from portia import Config, LogLevel, PlanRunState, Portia, example_tool_registry
from portia.cli import CLIExecutionHooks
from portia.end_user import EndUser

load_dotenv()

portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry,
)


# Simple Example
plan_run = portia.run(
    "Get the temperature in London and Sydney and then add the two temperatures",
)

# We can also provide additional data about the end user to the process
plan_run = portia.run(
    "Please tell me a joke that is customized to my favorite sport",
    end_user=EndUser(
        external_id="user_789",
        email="hello@portialabs.ai",
        additional_data={
            "favorite_sport": "football",
        },
    ),
)

# When we hit a clarification we can ask our end user for clarification then resume the process
plan_run = portia.run(
    "Get the temperature in <location> and London and then add the two temperatures",
)

# Fetch run
plan_run = portia.storage.get_plan_run(plan_run.id)
# Update clarifications
if plan_run.state == PlanRunState.NEED_CLARIFICATION:
    for c in plan_run.get_outstanding_clarifications():
        # Here you prompt the user for the response to the clarification
        # via whatever mechanism makes sense for your use-case.
        new_value = "Sydney"
        plan_run = portia.resolve_clarification(
            plan_run=plan_run,
            clarification=c,
            response=new_value,
        )
    portia.resume(plan_run)

# You can also pass in a clarification handler to manage clarifications
portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry,
    execution_hooks=CLIExecutionHooks(),
)
plan_run = portia.run(
    "Get the temperature in <location> and London and then add the two temperatures",
)

# You can pass inputs into a plan
plan_run = portia.run(
    "Get the temperature for the provided city", plan_run_inputs={"$city": "Lisbon"}
)
