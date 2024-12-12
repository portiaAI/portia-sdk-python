from abc import ABC

from portia.plan import Workflow


class WorkflowStorage(ABC):
    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow."""
        raise NotImplementedError("save_workflow is not implemented")

    def get_workflow(self, workflow_id: str) -> Workflow:
        """Retrieve a workflow by its ID."""
        raise NotImplementedError("get_workflow is not implemented")

    def match_similar_workflows(self, query: str) -> list[Workflow]:
        """Retrieve similar workflows by query"""
        raise NotImplementedError("match_similar_workflows is not implemented")

    def __add__(self, other: "WorkflowStorage") -> "WorkflowStorage":
        return AggregatedWorkflowStorage([self, other])
