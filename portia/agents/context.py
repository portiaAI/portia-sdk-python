"""Context builder that generates contextual information for the workflow.

This module defines a set of functions that build various types of context
required for the workflow execution. It takes information about inputs,
outputs, clarifications, and execution metadata to build context strings
used by the agent to perform tasks. The context can be extended with
additional system or user-provided data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from portia.clarification import ArgumentClarification, ClarificationListType

if TYPE_CHECKING:
    from portia.agents.base_agent import Output
    from portia.execution_context import ExecutionContext
    from portia.plan import Step, Variable
    from portia.workflow import Workflow


def generate_main_system_context(system_context_extensions: list[str] | None = None) -> list[str]:
    """Generate the main system context.

    Args:
        system_context_extensions (list[str] | None): Optional list of strings to extend
                                                     the system context.

    Returns:
        list[str]: A list of strings representing the system context.

    """
    system_context = [
        "System Context:",
        f"Today's date is {datetime.now(UTC).strftime('%Y-%m-%d')}",
    ]
    if system_context_extensions:
        system_context.extend(system_context_extensions)
    return system_context


def generate_input_context(
    inputs: list[Variable],
    previous_outputs: dict[str, Output],
) -> list[str]:
    """Generate context for the inputs and indicate which ones were used.

    Args:
        inputs (list[Variable]): The list of input variables for the current step.
        previous_outputs (dict[str, Output]): A dictionary of previous step outputs.

    Returns:
        list[str]: A list of strings representing the input context.

    """
    input_context = ["Inputs: the original inputs provided by the planner"]
    used_outputs = set()
    for var in inputs:
        if var.value is not None:
            input_context.extend(
                [
                    f"input_name: {var.name}",
                    f"input_value: {var.value}",
                    f"input_description: {var.description}",
                    "----------",
                ],
            )
        elif var.name in previous_outputs:
            input_context.extend(
                [
                    f"input_name: {var.name}",
                    f"input_value: {previous_outputs[var.name].value}",
                    f"input_description: {var.description}",
                    "----------",
                ],
            )
            used_outputs.add(var.name)

    unused_output_keys = set(previous_outputs.keys()) - used_outputs
    if len(unused_output_keys) > 0:
        input_context.append(
            "Broader context: This may be useful information from previous steps that can "
            "indirectly help you.",
        )
        for output_key in unused_output_keys:
            # We truncate the output value to 1000 characters to avoid overwhelming the
            # LLM with too much information.
            output_val = (str(previous_outputs[output_key].value) or "")[:1000]
            input_context.extend(
                [
                    f"output_name: {output_key}",
                    f"output_value: {output_val}",
                    "----------",
                ],
            )

    return input_context


def generate_clarification_context(clarifications: ClarificationListType, step: int) -> list[str]:
    """Generate context from clarifications for the given step.

    Args:
        clarifications (ClarificationListType): A list of clarification objects.
        step (int): The step index for which clarifications are being generated.

    Returns:
        list[str]: A list of strings representing the clarification context.

    """
    clarification_context = []
    # It's important we distinguish between clarifications for the current step where we really
    # want to use the value provided, and clarifications for other steps which may be useful
    # (e.g. consider a plan with 10 steps, each needing the same clarification, we don't want
    # to ask 10 times) but can also lead to side effects (e.g. consider a Plan with two steps where
    # both steps use different tools but with the same parameter name. We don't want to use the
    # clarification from the previous step for the second tool)
    current_step_clarifications = []
    other_step_clarifications = []

    for clarification in clarifications:
        if clarification.step == step:
            current_step_clarifications.append(clarification)
        else:
            other_step_clarifications.append(clarification)

    if current_step_clarifications:
        clarification_context.extend(
            [
                "Clarifications:",
                "This section contains the user provided response to previous clarifications",
                "for the current step. They should take priority over any other context given.",
            ],
        )
        for clarification in current_step_clarifications:
            if isinstance(clarification, (ArgumentClarification)) and clarification.step == step:
                clarification_context.extend(
                    [
                        f"input_name: {clarification.argument_name}",
                        f"clarification_reason: {clarification.user_guidance}",
                        f"input_value: {clarification.response}",
                        "----------",
                    ],
                )

    return clarification_context


def generate_context_from_execution_context(context: ExecutionContext) -> list[str]:
    """Generate context from the execution context.

    Args:
        context (ExecutionContext): The execution context containing metadata and additional data.

    Returns:
        list[str]: A list of strings representing the execution context.

    """
    if not context.end_user_id and not context.additional_data:
        return []

    execution_context = ["Metadata: This section contains general context about this execution."]
    if context.end_user_id:
        execution_context.extend(
            [
                f"end_user_id: {context.end_user_id}",
            ],
        )
    for key, value in context.additional_data.items():
        execution_context.extend(
            [
                f"context_key_name: {key} context_key_value: {value}",
                "----------",
            ],
        )
    return execution_context


def build_context(ctx: ExecutionContext, step: Step, workflow: Workflow) -> str:
    """Build the context string for the agent using inputs/outputs/clarifications/ctx.

    Args:
        ctx (ExecutionContext): The execution context containing agent and system metadata.
        step (Step): The current step in the workflow, including inputs.
        workflow (Workflow): The current workflow containing outputs and clarifications.

    Returns:
        str: A string containing all relevant context information.

    """
    inputs = step.inputs
    previous_outputs = workflow.outputs.step_outputs
    clarifications = workflow.outputs.clarifications

    system_context = generate_main_system_context(ctx.agent_system_context_extension)

    # exit early if no additional information
    if not inputs and not clarifications and not previous_outputs:
        return "\n".join(system_context)

    context = ["Additional context: You MUST use this information to complete your task."]

    # Generate and append input context
    input_context = generate_input_context(inputs, previous_outputs)
    context.extend(input_context)

    # Generate and append clarifications context
    clarification_context = generate_clarification_context(
        clarifications,
        workflow.current_step_index,
    )
    context.extend(clarification_context)

    # Handle execution context
    execution_context = generate_context_from_execution_context(ctx)
    context.extend(execution_context)

    # Append System Context
    context.extend(system_context)

    return "\n".join(context)
