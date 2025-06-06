"""Tests for the CLI clarification handler."""

from unittest.mock import MagicMock, patch

import click
import pytest
from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    UserVerificationClarification,
    ValueConfirmationClarification,
)
from portia.cli_clarification_handler import CLIClarificationHandler
from portia.prefixed_uuid import PlanRunUUID


@pytest.fixture
def cli_handler() -> CLIClarificationHandler:
    """Create a CLI clarification handler for testing."""
    return CLIClarificationHandler()


@patch("portia.cli_clarification_handler.click.echo")
def test_action_clarification(mock_echo: MagicMock, cli_handler: CLIClarificationHandler) -> None:
    """Test handling of action clarifications."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please authenticate",
        action_url=HttpUrl("https://example.com/auth"),
        source="Test cli clarification handler",
    )

    cli_handler.handle_action_clarification(clarification, on_resolution, on_error)

    # Verify echo was called with the expected message
    mock_echo.assert_called_once()
    echo_message = mock_echo.call_args[0][0]
    assert "Please authenticate" in click.unstyle(echo_message)
    assert "https://example.com/auth" in click.unstyle(echo_message)

    # Verify callbacks were not called
    on_resolution.assert_not_called()
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.echo")
@patch("portia.cli_clarification_handler.click.confirm")
def test_action_clarification_with_confirmation(
    mock_confirm: MagicMock,
    mock_echo: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of action clarifications."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_confirm.return_value = True

    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please authenticate",
        action_url=HttpUrl("https://example.com/auth"),
        require_confirmation=True,
        source="Test cli clarification handler",
    )

    cli_handler.handle_action_clarification(clarification, on_resolution, on_error)

    # Verify echo was called with the expected message
    mock_echo.assert_called_once()
    echo_message = mock_echo.call_args[0][0]
    assert "Please authenticate" in click.unstyle(echo_message)
    assert "https://example.com/auth" in click.unstyle(echo_message)

    mock_confirm.assert_called_once()
    confirm_message = mock_confirm.call_args[1]["text"]
    assert "Please confirm once the action is complete." in click.unstyle(confirm_message)

    # Verify resolution callback was called with True
    on_resolution.assert_called_once_with(clarification, True)  # noqa: FBT003
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.echo")
@patch("portia.cli_clarification_handler.click.confirm")
def test_action_clarification_with_confirmation_rejected(
    mock_confirm: MagicMock,
    mock_echo: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of action clarifications."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_confirm.return_value = False

    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please authenticate",
        action_url=HttpUrl("https://example.com/auth"),
        require_confirmation=True,
        source="Test cli clarification handler",
    )

    cli_handler.handle_action_clarification(clarification, on_resolution, on_error)

    # Verify echo was called with the expected message
    mock_echo.assert_called_once()
    echo_message = mock_echo.call_args[0][0]
    assert "Please authenticate" in click.unstyle(echo_message)
    assert "https://example.com/auth" in click.unstyle(echo_message)

    mock_confirm.assert_called_once()
    confirm_message = mock_confirm.call_args[1]["text"]
    assert "Please confirm once the action is complete." in click.unstyle(confirm_message)

    # Verify resolution callback was called with True
    on_resolution.assert_not_called()
    on_error.assert_called_once_with(clarification, "Clarification was rejected by the user")


@patch("portia.cli_clarification_handler.click.prompt")
def test_input_clarification(mock_prompt: MagicMock, cli_handler: CLIClarificationHandler) -> None:
    """Test handling of input clarifications."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_prompt.return_value = "user input"

    clarification = InputClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Enter your name",
        argument_name="name",
        source="Test cli clarification handler",
    )

    cli_handler.handle_input_clarification(clarification, on_resolution, on_error)

    # Verify prompt was called
    mock_prompt.assert_called_once()
    prompt_text = mock_prompt.call_args[0][0]
    assert "Enter your name" in click.unstyle(prompt_text)

    # Verify resolution callback was called with user input
    on_resolution.assert_called_once_with(clarification, "user input")
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.prompt")
def test_multiple_choice_clarification(
    mock_prompt: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of multiple choice clarifications."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_prompt.return_value = "option2"

    clarification = MultipleChoiceClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Choose a color",
        argument_name="color",
        options=["option1", "option2", "option3"],
        source="Test cli clarification handler",
    )

    cli_handler.handle_multiple_choice_clarification(clarification, on_resolution, on_error)

    # Verify prompt was called with choices
    mock_prompt.assert_called_once()
    prompt_text = mock_prompt.call_args[0][0]
    assert "Choose a color" in click.unstyle(prompt_text)

    # Verify type parameter was a click.Choice with the correct options
    choice_type = mock_prompt.call_args[1]["type"]
    assert isinstance(choice_type, click.Choice)
    assert choice_type.choices == ("option1", "option2", "option3")

    # Verify resolution callback was called with selected option
    on_resolution.assert_called_once_with(clarification, "option2")
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.confirm")
def test_value_confirmation_clarification_confirmed(
    mock_confirm: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of value confirmation clarifications when confirmed."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_confirm.return_value = True

    clarification = ValueConfirmationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Confirm deletion?",
        argument_name="confirm_delete",
        source="Test cli clarification handler",
    )

    cli_handler.handle_value_confirmation_clarification(clarification, on_resolution, on_error)

    # Verify confirm was called
    mock_confirm.assert_called_once()
    confirm_text = mock_confirm.call_args[1]["text"]
    assert "Confirm deletion?" in click.unstyle(confirm_text)

    # Verify resolution callback was called with True
    on_resolution.assert_called_once_with(clarification, True)  # noqa: FBT003
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.confirm")
def test_value_confirmation_clarification_rejected(
    mock_confirm: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of value confirmation clarifications when rejected."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_confirm.return_value = False

    clarification = ValueConfirmationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Confirm deletion?",
        argument_name="confirm_delete",
        source="Test cli clarification handler",
    )

    cli_handler.handle_value_confirmation_clarification(clarification, on_resolution, on_error)

    # Verify confirm was called
    mock_confirm.assert_called_once()

    # Verify error callback was called with rejection message
    on_resolution.assert_not_called()
    on_error.assert_called_once()
    assert "rejected" in on_error.call_args[0][1]


@patch("portia.cli_clarification_handler.click.echo")
@patch("portia.cli_clarification_handler.click.prompt")
def test_custom_clarification(
    mock_prompt: MagicMock,
    mock_echo: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of custom clarifications."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_prompt.return_value = "custom response"

    clarification = CustomClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Custom action needed",
        name="custom_action",
        data={"key1": "value1", "key2": "value2"},
        source="Test cli clarification handler",
    )

    cli_handler.handle_custom_clarification(clarification, on_resolution, on_error)

    # Verify echo was called twice (once for guidance, once for data)
    assert mock_echo.call_count == 2
    guidance_text = mock_echo.call_args_list[0][0][0]
    data_text = mock_echo.call_args_list[1][0][0]
    assert "Custom action needed" in click.unstyle(guidance_text)
    assert "key1" in click.unstyle(data_text)
    assert "value1" in click.unstyle(data_text)

    # Verify prompt was called
    mock_prompt.assert_called_once()

    # Verify resolution callback was called with user input
    on_resolution.assert_called_once_with(clarification, "custom response")
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.confirm")
def test_user_verification_clarification_confirmed(
    mock_confirm: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of user verification clarifications when confirmed."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_confirm.return_value = True

    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please verify this information",
        source="Test cli clarification handler",
    )

    cli_handler.handle(clarification, on_resolution, on_error)

    # Verify confirm was called
    mock_confirm.assert_called_once()
    confirm_text = mock_confirm.call_args[1]["text"]
    assert "Please verify this information" in click.unstyle(confirm_text)

    # Verify resolution callback was called with True
    on_resolution.assert_called_once_with(clarification, True)  # noqa: FBT003
    on_error.assert_not_called()


@patch("portia.cli_clarification_handler.click.confirm")
def test_user_verification_clarification_rejected(
    mock_confirm: MagicMock,
    cli_handler: CLIClarificationHandler,
) -> None:
    """Test handling of user verification clarifications when rejected."""
    on_resolution = MagicMock()
    on_error = MagicMock()

    mock_confirm.return_value = False

    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please verify this information",
        source="Test cli clarification handler",
    )

    cli_handler.handle(clarification, on_resolution, on_error)

    # Verify confirm was called
    mock_confirm.assert_called_once()
    # Verify resolution callback was called (even when user rejects)
    on_resolution.assert_called_once_with(clarification, False)  # noqa: FBT003
    # Verify error callback was NOT called (UserVerification always resolves)
    on_error.assert_not_called()
