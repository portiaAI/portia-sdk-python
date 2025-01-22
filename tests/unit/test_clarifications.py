"""Test simple agent."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    InputClarification,
    MultiChoiceClarification,
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


def test_value_multi_choice_validation() -> None:
    """Test clarifications error on invalid response."""
    clarification = MultiChoiceClarification(
        argument_name="test",
        user_guidance="test",
        options=["yes"],
    )
    with pytest.raises(ValueError):  # noqa: PT011
        clarification.resolve("this is  not the answer")
    clarification.resolve("yes")

    with pytest.raises(ValueError):  # noqa: PT011
        clarification = MultiChoiceClarification(
            argument_name="test",
            user_guidance="test",
            options=["yes"],
            resolved=True,
            response="No",
        )

    MultiChoiceClarification(
        argument_name="test",
        user_guidance="test",
        options=["yes"],
        resolved=True,
        response="yes",
    )
