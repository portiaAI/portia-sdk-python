"""Context builder that takes the current data from the workflow and generate context."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from portia.clarification import Clarification, InputClarification, MultiChoiceClarification

if TYPE_CHECKING:
    from portia.agents.base_agent import Output
    from portia.config import Config
    from portia.plan import Step, Variable
    from portia.workflow import Workflow


def generate_main_system_context(system_context_extensions: list[str] | None = None) -> list[str]:
    """Generate the main system context."""
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
    """Generate context for the inputs returning the context and which inputs were used."""
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
                    f"input_value: {previous_outputs[var.name]}",
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
            input_context.extend(
                [
                    f"output_name: {output_key}",
                    f"output_value: {previous_outputs[output_key]}",
                    "----------",
                ],
            )

    return input_context


def generate_clarification_context(clarifications: list[Clarification]) -> list[str]:
    """Generate context from clarifications."""
    clarification_context = []
    if clarifications:
        clarification_context.extend(
            [
                "Clarifications:",
                "This section contains the user provided response to previous clarifications",
                "They should take priority over any other context given.",
            ],
        )
        for clarification in clarifications:
            if isinstance(clarification, (InputClarification, MultiChoiceClarification)):
                clarification_context.extend(
                    [
                        f"input_name: {clarification.argument_name}",
                        f"clarification_reason: {clarification.user_guidance}",
                        f"input_value: {clarification.response}",
                        "----------",
                    ],
                )
    return clarification_context


def generate_metadata_context(metadata: dict[str, str]) -> list[str]:
    """Generate context from metadata."""
    metadata_context = ["Metadata: This section contains general metadata about this execution."]
    for key, value in metadata.items():
        metadata_context.extend(
            [
                f"metadata_name: {key}",
                f"metadata_value: {value}",
                "----------",
            ],
        )
    return metadata_context


def build_context(step: Step, workflow: Workflow, config: Config) -> str:
    """Turn inputs and past outputs into a context string for the agent."""
    inputs = step.inputs
    previous_outputs = workflow.step_outputs
    clarifications = workflow.clarifications
    metadata = workflow.metadata
    system_context = generate_main_system_context(config.agent_system_context_extension)

    # exit early if no additional information
    if not inputs and not clarifications and not previous_outputs:
        return "\n".join(system_context)

    context = ["Additional context: You MUST use this information to complete your task."]

    # Generate and append input context
    input_context = generate_input_context(inputs, previous_outputs)
    context.extend(input_context)

    # Generate and append clarifications context
    clarification_context = generate_clarification_context(clarifications)
    context.extend(clarification_context)

    # Handle metadata context
    metadata_context = generate_metadata_context(metadata)
    context.extend(metadata_context)

    # Append System Context
    context.extend(system_context)

    return "\n".join(context)
