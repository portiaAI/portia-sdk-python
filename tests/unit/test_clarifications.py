"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    ClarificationCategory,
    ClarificationUUID,
    CustomClarification,
    MultipleChoiceClarification,
    UserVerificationClarification,
)
from portia.prefixed_uuid import PlanRunUUID
from portia.storage import DiskFileStorage
from tests.utils import get_test_plan_run

if TYPE_CHECKING:
    from pathlib import Path


def test_action_clarification_ser() -> None:
    """Test action clarifications can be serialized."""
    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
        source="Test clarification",
    )
    clarification_model = clarification.model_dump()
    assert clarification_model["action_url"] == "https://example.com/"


def test_clarification_uuid_assign() -> None:
    """Test clarification assign correct UUIDs."""
    clarification = ActionClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
        source="Test clarification",
    )
    assert isinstance(clarification.id, ClarificationUUID)


def test_value_multi_choice_validation() -> None:
    """Test clarifications error on invalid response."""
    with pytest.raises(ValueError):  # noqa: PT011
        MultipleChoiceClarification(
            plan_run_id=PlanRunUUID(),
            argument_name="test",
            user_guidance="test",
            options=["yes"],
            resolved=True,
            response="No",
            source="Test clarification",
        )

    MultipleChoiceClarification(
        plan_run_id=PlanRunUUID(),
        argument_name="test",
        user_guidance="test",
        options=["yes"],
        resolved=True,
        response="yes",
        source="Test clarification",
    )


def test_custom_clarification_deserialize(tmp_path: Path) -> None:
    """Test clarifications error on invalid response."""
    (plan, plan_run) = get_test_plan_run()

    clarification_one = CustomClarification(
        plan_run_id=plan_run.id,
        user_guidance="Please provide data",
        name="My Clarification",
        data={"email": {"test": "hello@example.com"}},
        source="Test clarification",
    )

    storage = DiskFileStorage(storage_dir=str(tmp_path))

    plan_run.outputs.clarifications = [clarification_one]

    storage.save_plan(plan)
    storage.save_plan_run(plan_run)
    retrieved = storage.get_plan_run(plan_run.id)
    assert isinstance(retrieved.outputs.clarifications[0], CustomClarification)
    assert retrieved.outputs.clarifications[0].data == {"email": {"test": "hello@example.com"}}


def test_user_verification_clarification() -> None:
    """Test user verification clarification creation and serialization."""
    clarification = UserVerificationClarification(
        plan_run_id=PlanRunUUID(),
        user_guidance="Please verify this information",
        source="Test clarification",
    )

    # Verify category is set correctly
    assert clarification.category == ClarificationCategory.USER_VERIFICATION

    # Verify serialization
    clarification_model = clarification.model_dump()
    assert clarification_model["category"] == "User Verification"
    assert clarification_model["user_guidance"] == "Please verify this information"
    assert isinstance(clarification.id, ClarificationUUID)
