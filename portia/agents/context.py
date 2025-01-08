"""Context builder that takes the current data from the workflow and generate context."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from portia.clarification import Clarification, InputClarification, MultiChoiceClarification

if TYPE_CHECKING:
    from portia.agents.base_agent import Output
    from portia.plan import Variable


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
    input_context = []
    used_outputs = set()
    for var in inputs:
        if var.value is not None:
            input_context.extend(
                [
                    f"name: {var.name}",
                    f"value: {var.value}",
                    f"description: {var.description}",
                    "----------",
                ],
            )
        elif var.name in previous_outputs:
            input_context.extend(
                [
                    f"name: {var.name}",
                    f"value: {previous_outputs[var.name]}",
                    f"description: {var.description}",
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
                    f"name: {output_key}",
                    f"value: {previous_outputs[output_key]}",
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
                "Clarifications:"
                "This section contains user provided input that might be useful to help your task",
            ],
        )
        for clarification in clarifications:
            if isinstance(clarification, (InputClarification, MultiChoiceClarification)):
                clarification_context.extend(
                    [
                        f"argument: {clarification.argument_name}",
                        f"clarification reason: {clarification.user_guidance}",
                        f"value: {clarification.response}",
                        "----------",
                    ],
                )
    return clarification_context


def build_context(
    inputs: list[Variable],
    previous_outputs: dict[str, Output],
    clarifications: list[Clarification],
    system_context_extensions: list[str] | None = None,
) -> str:
    """Turn inputs and past outputs into a context string for the agent."""
    system_context = generate_main_system_context(system_context_extensions)

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

    # Append System Context
    context.extend(system_context)

    return "\n".join(context)
