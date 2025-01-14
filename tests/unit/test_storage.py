"""Test simple agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from portia.plan import Plan, PlanContext
from portia.storage import PlanStorage, WorkflowStorage
from portia.workflow import Workflow

if TYPE_CHECKING:
    from uuid import UUID


def test_storage_base_classes() -> None:
    """Test PlanStorage raises."""

    class MyStorage(WorkflowStorage, PlanStorage):
        """Override to test base."""

        def save_plan(self, plan: Plan) -> None:
            return super().save_plan(plan)  # type: ignore  # noqa: PGH003

        def get_plan(self, plan_id: UUID) -> Plan:
            return super().get_plan(plan_id)  # type: ignore  # noqa: PGH003

        def save_workflow(self, workflow: Workflow) -> None:
            return super().save_workflow(workflow)  # type: ignore  # noqa: PGH003

        def get_workflow(self, workflow_id: UUID) -> Workflow:
            return super().get_workflow(workflow_id)  # type: ignore  # noqa: PGH003

    storage = MyStorage()
    plan = Plan(plan_context=PlanContext(query="", tool_list=[]), steps=[])
    workflow = Workflow(
        plan_id=plan.id,
    )
    with pytest.raises(NotImplementedError):
        storage.save_plan(plan)

    with pytest.raises(NotImplementedError):
        storage.get_plan(plan.id)

    with pytest.raises(NotImplementedError):
        storage.save_workflow(workflow)

    with pytest.raises(NotImplementedError):
        storage.get_workflow(workflow.id)
