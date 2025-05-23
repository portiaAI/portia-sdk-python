"""Test clarification handler."""

from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    Clarification,
    ClarificationCategory,
    CustomClarification,
    InputClarification,
    MultipleChoiceClarification,
    UserVerificationClarification,
    ValueConfirmationClarification,
)
from portia.clarification_handler import ClarificationHandler
from portia.prefixed_uuid import PlanRunUUID


class TestClarificationHandler(ClarificationHandler):
    """Handles clarifications using mocks in this test."""


def test_action_clarification() -> None:
    """Test that ActionClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    on_resolution = MagicMock()
    on_error = MagicMock()
    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
        source="Test clarification handler",
    )

    # Test without implementation
    with pytest.raises(NotImplementedError):
        handler.handle(clarification, on_resolution, on_error)

    # Test with implementation
    handler.handle_action_clarification = MagicMock()
    handler.handle(clarification, on_resolution, on_error)
    handler.handle_action_clarification.assert_called_once_with(
        clarification,
        on_resolution,
        on_error,
    )


def test_input_clarification() -> None:
    """Test that InputClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    on_resolution = MagicMock()
    on_error = MagicMock()
    clarification = InputClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        argument_name="test",
        source="Test clarification handler",
    )

    # Test without implementation
    with pytest.raises(NotImplementedError):
        handler.handle(clarification, on_resolution, on_error)

    # Test with implementation
    handler.handle_input_clarification = MagicMock()
    handler.handle(clarification, on_resolution, on_error)
    handler.handle_input_clarification.assert_called_once_with(
        clarification,
        on_resolution,
        on_error,
    )


def test_multiple_choice_clarification() -> None:
    """Test that MultipleChoiceClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    on_resolution = MagicMock()
    on_error = MagicMock()
    clarification = MultipleChoiceClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        argument_name="test",
        options=["option1", "option2"],
        source="Test clarification handler",
    )

    # Test without implementation
    with pytest.raises(NotImplementedError):
        handler.handle(clarification, on_resolution, on_error)

    # Test with implementation
    handler.handle_multiple_choice_clarification = MagicMock()
    handler.handle(clarification, on_resolution, on_error)
    handler.handle_multiple_choice_clarification.assert_called_once_with(
        clarification,
        on_resolution,
        on_error,
    )


def test_value_confirmation_clarification() -> None:
    """Test that ValueConfirmationClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    on_resolution = MagicMock()
    on_error = MagicMock()
    clarification = ValueConfirmationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        argument_name="test",
        source="Test clarification handler",
    )

    # Test without implementation
    with pytest.raises(NotImplementedError):
        handler.handle(clarification, on_resolution, on_error)

    # Test with implementation
    handler.handle_value_confirmation_clarification = MagicMock()
    handler.handle(clarification, on_resolution, on_error)
    handler.handle_value_confirmation_clarification.assert_called_once_with(
        clarification,
        on_resolution,
        on_error,
    )


def test_user_verification_clarification() -> None:
    """Test that UserVerificationClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    on_resolution = MagicMock()
    on_error = MagicMock()
    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        source="Test clarification handler",
    )

    # Test without implementation
    with pytest.raises(NotImplementedError):
        handler.handle(clarification, on_resolution, on_error)

    # Test with implementation
    handler.handle_user_verification_clarification = MagicMock()
    handler.handle(clarification, on_resolution, on_error)
    handler.handle_user_verification_clarification.assert_called_once_with(
        clarification,
        on_resolution,
        on_error,
    )


def test_custom_clarification_routing() -> None:
    """Test that CustomClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    on_resolution = MagicMock()
    on_error = MagicMock()
    clarification = CustomClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        name="test",
        data={"key": "value"},
        source="Test clarification handler",
    )

    # Test without implementation
    with pytest.raises(NotImplementedError):
        handler.handle(clarification, on_resolution, on_error)

    # Test with implementation
    handler.handle_custom_clarification = MagicMock()
    handler.handle(clarification, on_resolution, on_error)
    handler.handle_custom_clarification.assert_called_once_with(
        clarification,
        on_resolution,
        on_error,
    )


def test_invalid_clarification() -> None:
    """Test that CustomClarification is routed to the correct handler method."""
    handler = TestClarificationHandler()

    class UnhandledClarification(Clarification):
        pass

    clarification = UnhandledClarification(
        category=ClarificationCategory.CUSTOM,
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        source="Test clarification handler",
    )

    on_resolution = MagicMock()
    on_error = MagicMock()
    with pytest.raises(ValueError):  # noqa: PT011
        handler.handle(clarification, on_resolution, on_error)
