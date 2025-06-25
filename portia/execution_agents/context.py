"""Context builder that generates contextual information for the PlanRun.

This module defines a set of functions that build various types of context
required for the run execution. It takes information about inputs,
outputs, clarifications, and execution metadata to build context strings
used by the agent to perform tasks. The context can be extended with
additional system or user-provided data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from portia.clarification import (
    ClarificationListType,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.common import Serializable

if TYPE_CHECKING:
    from portia.execution_agents.output import Output
    from portia.plan_run import PlanRun
    from portia.tool import ToolRunContext


def generate_main_system_context() -> list[str]:
    """Generate the main system context.

    Returns:
        list[str]: A list of strings representing the system context.

    """
    return [
        "System Context:",
        f"Today's date is {datetime.now(UTC).strftime('%Y-%m-%d')}",
    ]


class StepInput(BaseModel):
    """An input for a step being executed by an execution agent."""

    name: str
    value: Serializable | None
    description: str


def generate_input_context(
    step_inputs: list[StepInput],
    previous_outputs: dict[str, Output],
) -> list[str]:
    """Generate context for the inputs and indicate which ones were used.

    Args:
        step_inputs (list[StepInput]): The list of inputs for the current step.
        previous_outputs (dict[str, Output]): A dictionary of previous step outputs.

    Returns:
        list[str]: A list of strings representing the input context.

    """
    input_context = ["Inputs: the original inputs provided by the planning_agent"]
    used_outputs = set()
    for step_input in step_inputs:
        input_context.extend(
            [
                f"input_name: {step_input.name}",
                f"input_value: {step_input.value}",
                f"input_description: {step_input.description}",
                "----------",
            ],
        )
        used_outputs.add(step_input.name)

    unused_output_keys = set(previous_outputs.keys()) - used_outputs
    if len(unused_output_keys) > 0:
        input_context.append(
            "Broader context: This may be useful information from previous steps that can "
            "indirectly help you.",
        )
        for output_key in unused_output_keys:
            # We truncate the output value to 10000 characters to avoid overwhelming the
            # LLM with too much information.
            output_val = (str(previous_outputs[output_key].get_value()) or "")[:10000]
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
            if (
                isinstance(
                    clarification,
                    InputClarification
                    | MultipleChoiceClarification
                    | ValueConfirmationClarification,
                )
                and clarification.step == step
            ):
                clarification_context.extend(
                    [
                        f"input_name: {clarification.argument_name}",
                        f"clarification_reason: {clarification.user_guidance}",
                        f"input_value: {clarification.response}",
                        "----------",
                    ],
                )

    return clarification_context


def generate_context_from_run_context(context: ToolRunContext) -> list[str]:
    """Generate context from the execution context.

    Args:
        context (ExecutionContext): The execution context containing metadata and additional data.

    Returns:
        list[str]: A list of strings representing the execution context.

    """
    execution_context = ["Metadata: This section contains general context about this execution."]
    execution_context.extend(
        [
            "Details on the end user.",
            "You can use this information when the user mentions themselves (i.e send me an email)",
            "if no other information is provided in the task.",
            f"end_user_id:{context.end_user.external_id}",
            f"end_user_name:{context.end_user.name}",
            f"end_user_email:{context.end_user.email}",
            f"end_user_phone:{context.end_user.phone_number}",
            f"end_user_attributes:{json.dumps(context.end_user.additional_data)}",
            "----------",
        ],
    )
    return execution_context


def build_context(
    ctx: ToolRunContext,
    plan_run: PlanRun,
    step_inputs: list[StepInput],
) -> str:
    """Build the context string for the agent using inputs/outputs/clarifications/ctx.

    Args:
        ctx (ToolRunContext): The tool run context containing agent and system metadata.
        plan_run (PlanRun): The current run containing outputs and clarifications.
        step_inputs (list[StepInput]): The inputs for the current step.

    Returns:
        str: A string containing all relevant context information.

    """
    previous_outputs = plan_run.outputs.step_outputs
    clarifications = plan_run.outputs.clarifications

    system_context = generate_main_system_context()

    # exit early if no additional information
    if not step_inputs and not clarifications and not previous_outputs:
        return "\n".join(system_context)

    context = ["Additional context: You MUST use this information to complete your task."]

    # Generate and append input context
    input_context = generate_input_context(step_inputs, previous_outputs)
    context.extend(input_context)

    # Generate and append clarifications context
    clarification_context = generate_clarification_context(
        clarifications,
        plan_run.current_step_index,
    )
    context.extend(clarification_context)

    # Handle execution context
    execution_context = generate_context_from_run_context(ctx)
    context.extend(execution_context)

    # Append System Context
    context.extend(system_context)

    return "\n".join(context)
