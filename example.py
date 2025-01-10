"""Simple Example."""

from uuid import UUID

from portia.config import Config, LogLevel
from portia.context import execution_context
from portia.example_tools.registry import example_tool_registry
from portia.runner import Runner

runner = Runner(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tool_registry=example_tool_registry,
)

# Simple Example
workflow = runner.run_query(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)


# With Execution Context
with execution_context(end_user_id="123"):
    workflow = runner.run_query(
        "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
    )


# Resuming after a clarification
workflow_id = UUID("12cc58ae-bf1d-4a96-9680-ec9e0cb564e5")
workflow = runner.storage.get_workflow(workflow_id)

for c in workflow.get_outstanding_clarifications():
    c.resolve(response=None)

with execution_context(context=workflow.execution_context):
    runner.execute_workflow(workflow)
