"""Test simple agent."""

from __future__ import annotations

from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    InputClarification,
    ValueConfirmationClarification,
)


def test_clarification_resolve() -> None:
    """Test clarifications can be resolved."""
    clarification = InputClarification(
        argument_name="test",
        user_guidance="test",
    )
    clarification.resolve("res")
    assert clarification.resolved
    assert clarification.response == "res"


def test_value_confirmation_clarification_resolve() -> None:
    """Test clarifications can be resolved."""
    clarification = ValueConfirmationClarification(
        argument_name="test",
        user_guidance="test",
        response="this is the answer",
    )
    clarification.resolve("this is not the answer")
    assert clarification.resolved
    assert clarification.response == "this is the answer"


def test_action_clarification_ser() -> None:
    """Test action clarifications can be serialized."""
    clarification = ActionClarification(
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
    )
    clarification_model = clarification.model_dump()
    assert clarification_model["action_url"] == "https://example.com/"
