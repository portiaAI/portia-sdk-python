"""Clarification Handler.

This module defines the base ClarificationHandler class that determines how to handle clarifications
that arise during the execution of a workflow. It can be extended to customize the handling of
clarifications.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import click

from portia.clarification import (
    ActionClarification,
    Clarification,
    ClarificationCategory,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.logger import logger
from portia.workflow import Workflow, WorkflowState

if TYPE_CHECKING:
    from portia.runner import Runner


class ClarificationHandler:
    """Handles clarifications that arise during the execution of a workflow."""

    def handle(self, runner: Runner, workflow: Workflow, clarification: Clarification) -> Workflow:
        """Handle a clarification by routing it to the appropriate handler.

        Args:
            runner: The runner that is running the workflow
            workflow: The workflow that the clarification was raised on
            clarification: The clarification object to handle

        """
        match clarification.category:
            case ClarificationCategory.ACTION:
                return self.handle_action_clarification(
                    runner,
                    workflow,
                    cast(ActionClarification, clarification),
                )
            case ClarificationCategory.INPUT:
                return self.handle_input_clarification(
                    runner,
                    workflow,
                    cast(InputClarification, clarification),
                )
            case ClarificationCategory.MULTIPLE_CHOICE:
                return self.handle_multiple_choice_clarification(
                    runner,
                    workflow,
                    cast(MultipleChoiceClarification, clarification),
                )
            case ClarificationCategory.VALUE_CONFIRMATION:
                return self.handle_value_confirmation_clarification(
                    runner,
                    workflow,
                    cast(ValueConfirmationClarification, clarification),
                )
            case ClarificationCategory.CUSTOM:
                return self.handle_custom_clarification(
                    runner,
                    workflow,
                    cast(CustomClarification, clarification),
                )
            case ClarificationCategory.ARGUMENT:
                raise NotImplementedError("Argument clarification not implemented")

    def handle_action_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: ActionClarification,
    ) -> Workflow:
        """Handle a clarification that needs the user to complete an action (e.g. click a URL).

        Handles the action by showing the user the URL on the CLI and instructing them to click on
        it to proceed.
        """
        logger().info(
            f"{clarification.user_guidance} -- Please click on the link below to proceed.",
            f"{clarification.action_url}",
        )
        return runner.wait_for_ready(workflow)

    def handle_input_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: InputClarification,
    ) -> Workflow:
        """Handle a user input clarifications by asking the user for input from the CLI."""
        user_input = click.prompt(
            clarification.user_guidance + "\nPlease enter a value:\n",
        )
        return runner.resolve_clarification(clarification, user_input, workflow)

    def handle_multiple_choice_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: MultipleChoiceClarification,
    ) -> Workflow:
        """Handle a multi-choice clarification by asking the user for input from the CLI."""
        choices = click.Choice(clarification.options)
        user_input = click.prompt(
            clarification.user_guidance + "\nPlease choose a value:\n",
            type=choices,
        )
        return runner.resolve_clarification(clarification, user_input, workflow)

    def handle_value_confirmation_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: ValueConfirmationClarification,
    ) -> Workflow:
        """Handle a value confirmation clarification by asking the user to confirm from the CLI."""
        if click.confirm(text=clarification.user_guidance, default=False):
            return runner.resolve_clarification(
                clarification,
                response=True,
                workflow=workflow,
            )
        workflow.state = WorkflowState.FAILED
        runner.storage.save_workflow(workflow)
        return workflow

    def handle_custom_clarification(
        self,
        runner: Runner,
        workflow: Workflow,
        clarification: CustomClarification,
    ) -> Workflow:
        """Handle a custom clarification."""
        click.echo(clarification.user_guidance)
        click.echo(f"Additional data: {json.dumps(clarification.data)}")
        user_input = click.prompt("\nPlease enter a value:\n")
        return runner.resolve_clarification(clarification, user_input, workflow)
