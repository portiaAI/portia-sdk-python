* [Running in production](/running-in-production)
* Execution hooks

On this page

# Execution hooks

Execution hooks provide the ability for users to extend, intervene in or modify the running of agents in Portia, including allowing human control to be overlayed into the multi-agent plan run deterministically.
This is done by allowing you to run custom code at specific points in the agent run.
This can be very useful in a wide range of circumstances:

* To add human-in-the-loop verification before tool calls. For example, if you're building an agent to verify and then administer product refunds to customers, you likely want to include a human-in-the-loop check before the agent gives out the refund.
* To add guardrails into the system, causing the agent to exit if a bad output is given or skip steps if certain conditions are met
* To modify the args that tools are called with (for example to redact any leaked PII (personally identifiable information))
* To stream updates of the system to a frontend to display to the user
* To add custom logging

## Available hooks[​](#available-hooks "Direct link to Available hooks")

Portia provides several hook points where you can inject custom code:

* **Before plan run**: Run only before the first step of a plan run. This can be useful for any setup steps you need to take at the start of a run.
* **Before step**: Run before each step in the plan is executed. This can be useful if you want add a guardrail before each step.
* **After step**: Run after each step completes. This can be useful for streaming updates on each step to a frontend.
* **After last step**: Run only after the final step of a plan run. This can be useful for any cleanup steps that you need to take at the end of a run.
* **Before tool call**: Run directly before any tool is called. This can be useful if you want to add a human-in-the-loop check before running a particular tool, or if you want to add any checks on tool args before a tool is called. You can also alter the tool args in this hook if required.
* **After tool call**: Executed directly after any tool call completes. This can be useful if you want to add any guardrail that check tool output.

## Implementing an execution hook[​](#implementing-an-execution-hook "Direct link to Implementing an execution hook")

To implement a custom hook, simply define the code you want to run and then pass it in as a hook when creating your Portia instance.
We also provide several pre-made execution hooks that can be imported from `portia.execution_hooks`:

```
from dotenv import load_dotenv  
from portia import Plan, PlanRun, Portia, Step, logger  
from portia.execution_hooks import ExecutionHooks, clarify_on_all_tool_calls, log_step_outputs  
  
load_dotenv()  
  
def log_before_each_step(plan: Plan, plan_run: PlanRun, step: Step) -> None:  # noqa: ARG001  
    """Log the output of a step in the plan."""  
    logger().info(f"Running step with task {step.task} using tool {step.tool_id}")  
  
  
portia = Portia(  
    execution_hooks=ExecutionHooks(  
        # Out custom hook defined above  
        before_step_execution=log_before_each_step,  
        # Imported hook to raise a clarification before all tool calls  
        before_tool_call=clarify_on_all_tool_calls,  
        # Imported hook to log the result of all steps  
        after_step_execution=log_step_outputs,  
    ),  
)
```

### Human-in-the-loop checks[​](#human-in-the-loop-checks "Direct link to Human-in-the-loop checks")

In the 'before tool call' and 'after tool call' hooks, you can raise **clarifications** with the user if you require their input.
As with other clarifications, these clarifications are then handled by the user through your chosen clarification handler (see [the clarification docs ↗](/understand-clarifications). for more details).
This allows you to create powerful human-in-the-loop checks and guardrails.
As an example, the below code uses a `UserVerificationClarification` to ensure that the user verifies all calls to the refund tool.

```
from typing import Any  
from portia import Clarification, ClarificationCategory, ExecutionHooks, PlanRun, Portia, Step, Tool, ToolHardError  
from portia.clarification import UserVerificationClarification  
  
def clarify_before_refunds(  
    tool: Tool,  
    args: dict[str, Any],  
    plan_run: PlanRun,  
    step: Step,  
) -> Clarification | None:  
    # Only raise a clarification for the refund tool  
    if tool.id != "refund_tool":  
        return None  
  
    # Find if the clarification if we already raised it  
    previous_clarification = plan_run.get_clarification_for_step(ClarificationCategory.USER_VERIFICATION)  
  
    # If we haven't raised it, or it has been resolved, raise a clarification  
    if not previous_clarification or not previous_clarification.resolved:  
        return UserVerificationClarification(  
            plan_run_id=plan_run.id,  
            user_guidance=f"Are you happy to proceed with the call to {tool.name} with args {args}? "  
            "Enter 'y' or 'yes' to proceed",  
        )  
  
    # If the user didn't verify the tool call, error out  
    if str(previous_clarification.response).lower() not in ["y", "yes"]:  
        raise ToolHardError("User rejected tool call to {tool.name} with args {args}")  
  
    # If the user did verify the tool call, continue to the call  
    return None  
  
portia = Portia(execution_hooks=ExecutionHooks(before_tool_call=clarify_before_refunds))
```

This is a common use-case so you don't actually need to write this yourself - you can use our pre-made `clarify_on_tool_calls` hook.

Last updated on **Sep 9, 2025** by **github-actions[bot]**