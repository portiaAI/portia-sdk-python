"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import HttpUrl

from portia.clarification import (
    ActionClarification,
    ClarificationUUID,
    CustomClarification,
    MultipleChoiceClarification,
)
from portia.storage import DiskFileStorage
from portia.prefixed_uuid import WorkflowUUID
from tests.utils import get_test_workflow

if TYPE_CHECKING:
    from pathlib import Path


def test_action_clarification_ser() -> None:
    """Test action clarifications can be serialized."""
    clarification = ActionClarification(
        workflow_id=WorkflowUUID(),
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
    )
    clarification_model = clarification.model_dump()
    assert clarification_model["action_url"] == "https://example.com/"


def test_clarification_uuid_assign() -> None:
    """Test clarification assign correct UUIDs."""
    clarification = ActionClarification(
        workflow_id=WorkflowUUID(),
        user_guidance="test",
        action_url=HttpUrl("https://example.com"),
    )
    assert isinstance(clarification.id, ClarificationUUID)


def test_value_multi_choice_validation() -> None:
    """Test clarifications error on invalid response."""
    with pytest.raises(ValueError):  # noqa: PT011
        MultipleChoiceClarification(
            workflow_id=WorkflowUUID(),
            argument_name="test",
            user_guidance="test",
            options=["yes"],
            resolved=True,
            response="No",
        )

    MultipleChoiceClarification(
        workflow_id=WorkflowUUID(),
        argument_name="test",
        user_guidance="test",
        options=["yes"],
        resolved=True,
        response="yes",
    )


def test_custom_clarification_deserialize(tmp_path: Path) -> None:
    """Test clarifications error on invalid response."""
    (plan, workflow) = get_test_workflow()

    clarification_one = CustomClarification(
        workflow_id=workflow.id,
        user_guidance="Please provide data",
        name="My Clarification",
        data={"email": {"test": "hello@example.com"}},
    )

    storage = DiskFileStorage(storage_dir=str(tmp_path))

    workflow.outputs.clarifications = [clarification_one]

    storage.save_plan(plan)
    storage.save_workflow(workflow)
    retrieved = storage.get_workflow(workflow.id)
    assert isinstance(retrieved.outputs.clarifications[0], CustomClarification)
    assert retrieved.outputs.clarifications[0].data == {"email": {"test": "hello@example.com"}}
