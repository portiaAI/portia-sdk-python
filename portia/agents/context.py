"""Context builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.clarification import Clarification, InputClarification, MultiChoiceClarification

if TYPE_CHECKING:
    from portia.agents.base_agent import Output
    from portia.plan import Variable


def build_context(
    inputs: list[Variable],
    previous_outputs: dict[str, Output],
    clarifications: list[Clarification] | None = None,
    system_context: list[str] | None = None,
) -> str:
    """Turn inputs and past outputs into a context string for the agent."""
    inputs = inputs or []
    system_context = system_context or []
    if not inputs and not clarifications and not previous_outputs and not system_context:
        return "No additional context"

    context = "Additional context: You MUST use this information to complete your task.\n"
    used_outputs = set()
    for var in inputs:
        if var.value is not None:
            context += (
                f"name: {var.name}\nvalue: {var.value}\n"
                f"description: {var.description}\n\n----------\n\n"
            )
        elif var.name in previous_outputs:
            context += (
                f"name: {var.name}\nvalue: {previous_outputs[var.name]}\n"
                f"description: {var.description}\n\n----------\n\n"
            )
            used_outputs.add(var.name)

    if clarifications:
        context += (
            "Clarifications: This section contains user provided clarifications"
            " that might be useful to complete your task.\n"
        )
        for clarification in clarifications:
            if isinstance(clarification, (InputClarification, MultiChoiceClarification)):
                context += f"argument: {clarification.argument_name}\n"
                context += f"clarification reason: {clarification.user_guidance}\n"
                context += f"value: {clarification.response}\n\n----------\n\n"

    unused_output_keys = set(previous_outputs.keys()) - used_outputs
    if len(unused_output_keys) > 0:
        general_context = (
            "\nBroader context: This may be useful information from previous steps that can "
            "indirectly help you.\n"
        )
        unused_context = [
            f"name: {output_key}\nvalue: {previous_outputs[output_key]}\n\n----------\n\n"
            for output_key in unused_output_keys
        ]
        context += general_context + "\n".join(unused_context)
    if system_context:
        context += "\nSystem Context:\n"
        context += "\n".join(system_context)
        context += "\n\n----------\n\n"
    return context
