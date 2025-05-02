"""Simple Example."""

from dotenv import load_dotenv

from portia import (
    Config,
    LogLevel,
    Portia,
)
from portia.config import FEATURE_FLAG_AGENT_MEMORY_ENABLED
from portia.execution_agents.output import LocalDataValue
from portia.model import LLMProvider
from portia.plan import PlanInput

load_dotenv()
portia = Portia(
    Config.from_default(
        default_log_level=LogLevel.DEBUG,
        feature_flags={
            FEATURE_FLAG_AGENT_MEMORY_ENABLED: True,
        },
        llm_provider=LLMProvider.ANTHROPIC,
    ),
)
#
#
## Simple Example
# plan_run = portia.run(
#    "Get the temperature in London and Sydney and then add the two temperatures rounded to 2DP",
# )
#
## We can also provide additional data about the end user to the process
#
# plan_run = portia.run(
#    "Please tell me a joke that is customized to my favorite sport",
#    end_user=EndUser(
#        external_id="user_789",
#        email="hello@portialabs.ai",
#        additional_data={
#            "favorite_sport": "football",
#        },
#    ),
# )
#
## When we hit a clarification we can ask our end user for clarification then resume the process
## Deliberate typo in the second place name to hit the clarification
# plan_run = portia.run(
#    "Get the temperature in London and xydwne and then add the two temperatures rounded to 2DP",
# )
#
## Fetch run
# plan_run = portia.storage.get_plan_run(plan_run.id)
## Update clarifications
# if plan_run.state == PlanRunState.NEED_CLARIFICATION:
#    #for c in plan_run.get_outstanding_clarifications():
#        # Here you prompt the user for the response to the clarification
#        # via whatever mechanism makes sense for your use-case.
#        new_value = "Sydney"
#        plan_run = portia.resolve_clarification(
#            plan_run=plan_run,
#            clarification=c,
#            response=new_value,
#        )
#
## Execute again with the same execution context
# with execution_context(context=plan_run.execution_context):
#    portia.resume(plan_run)
#
## You can also pass in a clarification handler to manage clarifications
# portia = Portia(
#    Config.from_default(default_log_level=LogLevel.DEBUG),
#    tools=example_tool_registry,
#    execution_hooks=CLIExecutionHooks(),
# )
# plan_run = portia.run(
#    "Get the temperature in London and xydwne and then add the two temperatures rounded to 2DP",
# )

# You can pass inputs into a plan
plan_run = portia.run(
    "Download Large I/O + Pagination - Initial Designs from Drive, "
    "email it to me, then just take the first 100 words, then email it to me, then summarise it, then email it to me. My email is robbie@portialabs.ai. "
    "For the titleemail titles, use 'email 1' then 2, then 3 etc. and for the bodies, just include the content mentioned above."
)


portia = Portia(Config.from_default(large_output_threshold_tokens=10000))

# Specify the inputs you will use in the plan
plan_input = PlanInput(name="$city", description="The city to get the temperature for")
plan = portia.plan("Get the temperature for the provided city", plan_inputs=[plan_input])

# Specify the values for those inputs when you run the plan
plan_run_inputs = {plan_input: "London"}
plan_run = portia.run("Get the temperature for the provided city", plan_run_inputs=plan_run_inputs)

plan_run_inputs = {
    PlanInput(name="$city", description="The city to get the temperature for"): LocalDataValue(
        value="Lisbon"
    ),
}
plan_run = portia.run("Get the temperature for the provided city", plan_run_inputs=plan_run_inputs)
