"""Simple Example."""

from portia.config import Config, LogLevel, StorageClass
from portia.context import execution_context
from portia.open_source_tools.registry import example_tool_registry
from portia.runner import Runner
from portia.tool_registry import PortiaToolRegistry
from portia.workflow import WorkflowState

my_config = Config.from_default(
    default_log_level=LogLevel.DEBUG,
    storage_class=StorageClass.CLOUD,
)

tool_registry = example_tool_registry + PortiaToolRegistry(my_config)

runner = Runner(
    my_config,
    tool_registry=tool_registry,
)


# Simple Example
workflow = runner.execute_query(
    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
)

# We can also provide additional execution context to the process
with execution_context(end_user_id="123", additional_data={"email_address": "hello@portialabs.ai"}):
    plan = runner.execute_query(
        "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
    )

# When we hit a clarification we can ask our end user for clarification then resume the process
with execution_context(end_user_id="123", additional_data={"email_address": "hello@portialabs.ai"}):
    workflow = runner.execute_query(
        "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
    )


# Fetch workflow
workflow = runner.storage.get_workflow(workflow.id)
# Update clarifications
if workflow.state == WorkflowState.NEED_CLARIFICATION:
    for c in workflow.get_outstanding_clarifications():
        c.resolve(response=None)

    # Execute again with the same execution context
    with execution_context(context=workflow.execution_context):
        runner.execute_workflow(workflow)
