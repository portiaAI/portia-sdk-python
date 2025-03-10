"""Simple Example."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from portia import (
    Config,
    LogLevel,
    PlanRunState,
    Portia,
    example_tool_registry,
    execution_context,
)
from portia.cli import CLIExecutionHooks
from portia.config import StorageClass


load_dotenv(override=True)

config = Config.from_default(default_log_level=LogLevel.DEBUG, storage_class=StorageClass.CLOUD)

portia = Portia(
    config,
    tools=example_tool_registry,
)

async def run_multiple_queries():
    """Run multiple queries asynchronously using a thread pool."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        plan = portia.plan(
            "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
        )
        
        for i in range  (5):
            with execution_context(end_user_id=f"user_{i}", additional_data={"email_address": f"user_{i}@example.com"}):
                plan_run = portia._create_plan_run(
                    plan=plan,
                )
            task = loop.run_in_executor(
                executor,
                portia.resume,
                plan_run,
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results


asyncio.run(run_multiple_queries())

# # We can also provide additional execution context to the process
# with execution_context(end_user_id="123", additional_data={"email_address": "hello@portialabs.ai"}):
#     plan_run = portia.run(
#         "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
#     )

# # When we hit a clarification we can ask our end user for clarification then resume the process
# with execution_context(end_user_id="123", additional_data={"email_address": "hello@portialabs.ai"}):
#     # Deliberate typo in the second place name to hit the clarification
#     plan_run = portia.run(
#         "Get the temperature in London and xydwne and then add the two temperatures rounded to 2DP",
#     )

# # Fetch run
# plan_run = portia.storage.get_plan_run(plan_run.id)
# # Update clarifications
# if plan_run.state == PlanRunState.NEED_CLARIFICATION:
#     for c in plan_run.get_outstanding_clarifications():
#         # Here you prompt the user for the response to the clarification
#         # via whatever mechanism makes sense for your use-case.
#         new_value = "Sydney"
#         plan_run = portia.resolve_clarification(
#             plan_run=plan_run,
#             clarification=c,
#             response=new_value,
#         )

# # Execute again with the same execution context
# with execution_context(context=plan_run.execution_context):
#     portia.resume(plan_run)

# # You can also pass in a clarification handler to manage clarifications
# portia = Portia(
#     Config.from_default(default_log_level=LogLevel.DEBUG),
#     tools=example_tool_registry,
#     execution_hooks=CLIExecutionHooks(),
# )
# plan_run = portia.run(
#     "Get the temperature in London and xydwne and then add the two temperatures rounded to 2DP",
# )
