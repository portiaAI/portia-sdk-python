"""Simple Example."""

from portia.config import Config, LogLevel
from portia.context import execution_context
from portia.example_tools.registry import example_tool_registry
from portia.runner import Runner
from portia.workflow import WorkflowState

runner = Runner(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tool_registry=example_tool_registry,
)

# Simple Example
workflow = runner.create_workflow_from_query(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)
runner.execute_workflow(workflow)


# With Execution Context
with execution_context(end_user_id="123"):
    workflow = runner.create_workflow_from_query(
        "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
    )
    runner.execute_workflow()


# Resuming after a clarification
workflow = runner.storage.get_workflow(workflow.id)
if workflow.state == WorkflowState.NEED_CLARIFICATION:
    for c in workflow.get_outstanding_clarifications():
        c.resolve(response=None)

    with execution_context(context=workflow.execution_context):
        runner.execute_workflow(workflow)
