"""CLI Clarification Handler.

This module provides a CLI-specific implementation of the ClarificationHandler
that handles clarifications via the command-line interface.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from portia.clarification_handler import ClarificationHandler

if TYPE_CHECKING:
    from collections.abc import Callable

    from portia.clarification import (
        ActionClarification,
        Clarification,
        CustomClarification,
        InputClarification,
        MultipleChoiceClarification,
        UserVerificationClarification,
        ValueConfirmationClarification,
    )


class CLIClarificationHandler(ClarificationHandler):
    """Handles clarifications by obtaining user input from the CLI."""

    def handle_action_clarification(
        self,
        clarification: ActionClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle an action clarification.

        Does this by showing the user the URL on the CLI and instructing them to click on
        it to proceed.
        """
        click.echo(
            click.style(
                f"{clarification.user_guidance} -- Please click on the link below to proceed. "
                f"{clarification.action_url}",
                fg=87,
            ),
        )
        if clarification.require_confirmation:
            if click.confirm(
                text=click.style("Please confirm once the action is complete.", fg=87),
                default=False,
            ):
                on_resolution(clarification, True)  # noqa: FBT003
            else:
                on_error(clarification, "Clarification was rejected by the user")

    def handle_input_clarification(
        self,
        clarification: InputClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a user input clarifications by asking the user for input from the CLI."""
        user_input = click.prompt(
            click.style(clarification.user_guidance + "\nPlease enter a value", fg=87),
        )
        return on_resolution(clarification, user_input)

    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a multi-choice clarification by asking the user for input from the CLI."""
        choices = click.Choice(clarification.options)
        user_input = click.prompt(
            click.style(clarification.user_guidance + "\nPlease choose a value:\n", fg=87),
            type=choices,
        )
        return on_resolution(clarification, user_input)

    def handle_value_confirmation_clarification(
        self,
        clarification: ValueConfirmationClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        """Handle a value confirmation clarification by asking the user to confirm from the CLI."""
        if click.confirm(text=click.style(clarification.user_guidance, fg=87), default=False):
            on_resolution(clarification, True)  # noqa: FBT003
        else:
            on_error(clarification, "Clarification was rejected by the user")

    def handle_user_verification_clarification(
        self,
        clarification: UserVerificationClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a user verification clarification by asking the user to confirm from the CLI."""
        result = click.confirm(text=click.style(clarification.user_guidance, fg=87), default=False)
        on_resolution(clarification, result)

    def handle_custom_clarification(
        self,
        clarification: CustomClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle a custom clarification."""
        click.echo(click.style(clarification.user_guidance, fg=87))
        click.echo(click.style(f"Additional data: {json.dumps(clarification.data)}", fg=87))
        user_input = click.prompt(click.style("\nPlease enter a value", fg=87))
        return on_resolution(clarification, user_input)
