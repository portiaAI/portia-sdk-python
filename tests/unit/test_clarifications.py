"""Test simple agent."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    ClarificationUUID,
    MultipleChoiceClarification,
)


def test_action_clarification_ser() -> None:
    """Test action clarifications can be serialized."""
    clarification = ActionClarification(
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
    )
    clarification_model = clarification.model_dump()
    assert clarification_model["action_url"] == "https://example.com/"


def test_clarification_uuid_assign() -> None:
    """Test clarification assign correct UUIDs."""
    clarification = ActionClarification(
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
    )
    assert isinstance(clarification.id, ClarificationUUID)


def test_value_multi_choice_validation() -> None:
    """Test clarifications error on invalid response."""
    with pytest.raises(ValueError):  # noqa: PT011
        MultipleChoiceClarification(
            argument_name="test",
            user_guidance="test",
            options=["yes"],
            resolved=True,
            response="No",
        )

    MultipleChoiceClarification(
        argument_name="test",
        user_guidance="test",
        options=["yes"],
        resolved=True,
        response="yes",
    )
