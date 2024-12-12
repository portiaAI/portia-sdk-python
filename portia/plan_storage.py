from abc import ABC
from uuid import UUID

from portia.plan import Plan, Workflow


class PlanNotFoundError(Exception):
    pass


class PlanStorage(ABC):
    def save_plan(self, plan: Plan) -> None:
        """Save a plan."""
        raise NotImplementedError("save_plan is not implemented")

    def get_plan(self, plan_id: UUID) -> Plan:
        """Retrieve a plan by its ID."""
        raise NotImplementedError("get_plan is not implemented")


class WorkflowNotFoundError(Exception):
    pass


class WorkflowStorage(ABC):
    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow."""
        raise NotImplementedError("save_workflow is not implemented")

    def get_workflow(self, workflow_id: UUID) -> Workflow:
        """Retrieve a workflow by its ID."""
        raise NotImplementedError("get_workflow is not implemented")


class Storage(PlanStorage, WorkflowStorage):
    pass


class InMemoryStorage(Storage):
    plans: dict[UUID, Plan] = {}
    workflows: dict[UUID, Workflow] = {}

    def save_plan(self, plan: Plan) -> None:
        self.plans[plan.id] = plan

    def get_plan(self, plan_id: UUID) -> Plan:
        if plan_id in self.plans:
            return self.plans[plan_id]
        raise PlanNotFoundError

    def save_workflow(self, workflow: Workflow) -> None:
        self.workflows[workflow.id] = workflow

    def get_workflow(self, workflow_id: UUID) -> Workflow:
        if workflow_id in self.workflows:
            return self.workflows[workflow_id]
        raise WorkflowNotFoundError
