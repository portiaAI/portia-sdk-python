"""Simple Example."""

from dotenv import load_dotenv

from portia import (
    Config,
    FileReaderTool,
    InMemoryToolRegistry,
    LogLevel,
    PlanRunState,
    Portia,
    example_tool_registry,
)
from portia.end_user import EndUser

load_dotenv()

portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry + InMemoryToolRegistry.from_local_tools([FileReaderTool()]),
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
# There are two poem.txt files in the repo, so we get a clarification to select the correct one
plan_run = portia.run(
    "Read the poem.txt file and write a review of it",
)
# Update clarifications
if plan_run.state == PlanRunState.NEED_CLARIFICATION:
    for c in plan_run.get_outstanding_clarifications():
        # Here you prompt the user for the response to the clarification
        # via whatever mechanism makes sense for your use-case.
        new_value = "data/laser-sharks-ballad/poem.txt"
        plan_run = portia.resolve_clarification(
            plan_run=plan_run,
            clarification=c,
            response=new_value,
        )
    portia.resume(plan_run)

# You can also pass in a clarification handler to manage clarifications
portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry + InMemoryToolRegistry.from_local_tools([FileReaderTool()]),
)
plan_run = portia.run(
    "Read the poem.txt file and write a review of it",
)

# You can pass inputs into a plan
plan_run = portia.run(
    "Get the temperature for the provided city", plan_run_inputs={"$city": "Lisbon"}
)
