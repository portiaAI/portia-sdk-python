"""Storage classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from portia.errors import PlanNotFoundError, WorkflowNotFoundError
from portia.plan import Plan
from portia.workflow import Workflow

if TYPE_CHECKING:
    from uuid import UUID

    from portia.config import Config

T = TypeVar("T", bound=BaseModel)


class PlanStorage(ABC):
    """Base class for storing plans."""

    @abstractmethod
    def save_plan(self, plan: Plan) -> None:
        """Save a plan."""
        raise NotImplementedError("save_plan is not implemented")

    @abstractmethod
    def get_plan(self, plan_id: UUID) -> Plan:
        """Retrieve a plan by its ID."""
        raise NotImplementedError("get_plan is not implemented")


class WorkflowStorage(ABC):
    """Base class for storing plans."""

    @abstractmethod
    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow."""
        raise NotImplementedError("save_workflow is not implemented")

    @abstractmethod
    def get_workflow(self, workflow_id: UUID) -> Workflow:
        """Retrieve a workflow by its ID."""
        raise NotImplementedError("get_workflow is not implemented")


class Storage(PlanStorage, WorkflowStorage):
    """Combined base class for Plan + Workflow storage."""


class InMemoryStorage(Storage):
    """Simple storage class that keeps plans + workflows in memory."""

    plans: ClassVar[dict[UUID, Plan]] = {}
    workflows: ClassVar[dict[UUID, Workflow]] = {}

    def save_plan(self, plan: Plan) -> None:
        """Add plan to dict."""
        self.plans[plan.id] = plan

    def get_plan(self, plan_id: UUID) -> Plan:
        """Get plan from dict."""
        if plan_id in self.plans:
            return self.plans[plan_id]
        raise PlanNotFoundError(plan_id)

    def save_workflow(self, workflow: Workflow) -> None:
        """Add workflow to dict."""
        self.workflows[workflow.id] = workflow

    def get_workflow(self, workflow_id: UUID) -> Workflow:
        """Get workflow from dict."""
        if workflow_id in self.workflows:
            return self.workflows[workflow_id]
        raise WorkflowNotFoundError(workflow_id)


class DiskFileStorage(Storage):
    """Disk-based implementation of the Storage interface.

    Stores serialized Plan and Workflow objects as JSON files on disk.
    """

    def __init__(self, storage_dir: str | None) -> None:
        """Set storage dir."""
        self.storage_dir = storage_dir or ".portia"

    def _ensure_storage(self) -> None:
        """Ensure that the storage directory exists."""
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)

    def _write(self, file_name: str, content: BaseModel) -> None:
        """Write a serialized Plan or Workflow to a JSON file.

        Args:
            file_name (str): Name of the file.
            content (Union[Plan, Workflow]): The Plan or Workflow object to serialize.

        """
        self._ensure_storage()  # Ensure storage directory exists
        with Path(self.storage_dir, file_name).open("w") as file:
            file.write(content.model_dump_json(indent=4))

    def _read(self, file_name: str, model: type[T]) -> T:
        """Read a JSON file and deserialize it into a BaseModel instance.

        Args:
            file_name (str): Name of the file.
            model (type[BaseModel]): The model class to deserialize into.

        Returns:
            BaseModel: The deserialized model instance.

        """
        with Path(self.storage_dir, file_name).open("r") as file:
            f = file.read()
            return model.model_validate_json(f)

    def save_plan(self, plan: Plan) -> None:
        """Save a Plan object to the storage.

        Args:
            plan (Plan): The Plan object to save.

        """
        self._write(f"plan-{plan.id}.json", plan)

    def get_plan(self, plan_id: UUID) -> Plan:
        """Retrieve a Plan object by its ID.

        Args:
            plan_id (UUID): The ID of the Plan to retrieve.

        Returns:
            Plan: The retrieved Plan object.

        Raises:
            PlanNotFoundError: If the Plan is not found or validation fails.

        """
        try:
            return self._read(f"plan-{plan_id}.json", Plan)
        except (ValidationError, FileNotFoundError) as e:
            raise PlanNotFoundError(plan_id) from e

    def save_workflow(self, workflow: Workflow) -> None:
        """Save a Workflow object to the storage.

        Args:
            workflow (Workflow): The Workflow object to save.

        """
        self._write(f"workflow-{workflow.id}.json", workflow)

    def get_workflow(self, workflow_id: UUID) -> Workflow:
        """Retrieve a Workflow object by its ID.

        Args:
            workflow_id (UUID): The ID of the Workflow to retrieve.

        Returns:
            Workflow: The retrieved Workflow object.

        Raises:
            WorkflowNotFoundError: If the Workflow is not found or validation fails.

        """
        try:
            return self._read(f"workflow-{workflow_id}.json", Workflow)
        except (ValidationError, FileNotFoundError) as e:
            raise WorkflowNotFoundError(workflow_id) from e


class PortiaCloudStorage(Storage):
    """Save plans and workflows to portia cloud."""

    def __init__(self, config: Config) -> None:
        """Store tools in a tool set for easy access."""
        self.api_key = config.must_get_api_key("portia_api_key")
        self.api_endpoint = config.must_get("portia_api_endpoint", str)

    def save_plan(self, plan: Plan) -> None:
        """Add plan to cloud."""
        response = httpx.post(
            url=f"{self.api_endpoint}/api/v0/plans/",
            json={"id": str(plan.id), "json": plan.model_dump(mode="json")},
            headers={
                "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        print(response.content)
        response.raise_for_status()

    def get_plan(self, plan_id: UUID) -> Plan:
        """Get plan from cloud."""
        response = httpx.get(
            url=f"{self.api_endpoint}/api/v0/plans/{plan_id}/",
            headers={
                "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return Plan.model_validate(response.json()["json"])

    def save_workflow(self, workflow: Workflow) -> None:
        """Add workflow to cloud."""
        response = httpx.post(
            url=f"{self.api_endpoint}/api/v0/workflows/",
            json={
                "id": str(workflow.id),
                "json": workflow.model_dump(mode="json"),
                "plan": str(workflow.plan_id),
            },
            headers={
                "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        # If the workflow exists, update it instead
        if "workflow with this id already exists." in str(response.content):
            response = httpx.patch(
                url=f"{self.api_endpoint}/api/v0/workflows/{workflow.id}/",
                json={
                    "id": str(workflow.id),
                    "json": workflow.model_dump(mode="json"),
                    "plan": str(workflow.plan_id),
                },
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()

    def get_workflow(self, workflow_id: UUID) -> Workflow:
        """Get workflow from cloud."""
        response = httpx.get(
            url=f"{self.api_endpoint}/api/v0/workflows/{workflow_id}/",
            headers={
                "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return Workflow.model_validate(response.json()["json"])
