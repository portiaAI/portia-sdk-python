"""Storage classes for managing the saving and retrieval of plans, workflows, and tool calls.

This module defines a set of storage classes that provide different backends for saving, retrieving,
and managing plans, workflows, and tool calls. These storage classes include both in-memory and
file-based storage, as well as integration with the Portia Cloud API. Each class is responsible
for handling interactions with its respective storage medium, including validating responses
and raising appropriate exceptions when necessary.

Classes:
    - Storage (Base Class): A base class that defines common interfaces for all storage types,
    ensuring consistent methods for saving and retrieving plans, workflows, and tool calls.
    - InMemoryStorage: An in-memory implementation of the `Storage` class for storing plans,
    workflows, and tool calls in a temporary, volatile storage medium.
    - FileStorage: A file-based implementation of the `Storage` class for storing plans, workflows,
      and tool calls as local files in the filesystem.
    - PortiaCloudStorage: A cloud-based implementation of the `Storage` class that interacts with
    the Portia Cloud API to save and retrieve plans, workflows, and tool call records.

Each storage class handles the following tasks:
    - Sending and receiving data to its respective storage medium - memory, file system, or API.
    - Validating responses from storage and raising errors when necessary.
    - Handling exceptions and re-raising them as custom `StorageError` exceptions to provide
    more informative error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel, ValidationError

from portia.errors import PlanNotFoundError, StorageError, WorkflowNotFoundError
from portia.execution_context import ExecutionContext
from portia.logger import logger
from portia.plan import Plan, PlanContext, Step, PlanUUID
from portia.tool_call import ToolCallRecord, ToolCallStatus
from portia.workflow import (
    Workflow,
    WorkflowOutputs,
    WorkflowState,
    WorkflowUUID,
    WORKFLOW_UUID_PREFIX,
)

if TYPE_CHECKING:
    from portia.config import Config

T = TypeVar("T", bound=BaseModel)


class PlanStorage(ABC):
    """Abstract base class for storing and retrieving plans.

    Subclasses must implement the methods to save and retrieve plans.

    Methods:
        save_plan(self, plan: Plan) -> None:
            Save a plan.
        get_plan(self, plan_id: PlanUUID) -> Plan:
            Get a plan by ID.

    """

    @abstractmethod
    def save_plan(self, plan: Plan) -> None:
        """Save a plan.

        Args:
            plan (Plan): The Plan object to save.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("save_plan is not implemented")

    @abstractmethod
    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Retrieve a plan by its ID.

        Args:
            plan_id (PlanUUID): The UUID of the plan to retrieve.

        Returns:
            Plan: The Plan object associated with the provided plan_id.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("get_plan is not implemented")


class WorkflowStorage(ABC):
    """Abstract base class for storing and retrieving workflows.

    Subclasses must implement the methods to save and retrieve workflows.

    Methods:
        save_workflow(self, workflow: Workflow) -> None:
            Save a workflow.
        get_workflow(self, workflow_id: WorkflowUUID) -> Workflow:
            Get a workflow by ID.
        get_workflows(self, workflow_state: WorkflowState | None = None) -> list[Workflow]:
            Return workflows that match the given workflow_state

    """

    @abstractmethod
    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow.

        Args:
            workflow (Workflow): The Workflow object to save.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("save_workflow is not implemented")

    @abstractmethod
    def get_workflow(self, workflow_id: WorkflowUUID) -> Workflow:
        """Retrieve a workflow by its ID.

        Args:
            workflow_id (WorkflowUUID): The UUID of the workflow to retrieve.

        Returns:
            Workflow: The Workflow object associated with the provided workflow_id.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("get_workflow is not implemented")

    @abstractmethod
    def get_workflows(self, workflow_state: WorkflowState | None = None) -> list[Workflow]:
        """List workflows by their state.

        Args:
            workflow_state (WorkflowState | None): Optionally filter workflows by their state.

        Returns:
            list[Workflow]: A list of Workflow objects that match the given state.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("get_workflows is not implemented")


class ToolCallStorage(ABC):
    """Abstract base class for storing tool_calls.

    Subclasses must implement the method to save a tool_call.

    Methods:
        save_tool_call(self, tool_call: ToolCallRecord) -> None:
            Save a tool_call.

    """

    @abstractmethod
    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Save a ToolCall.

        Args:
            tool_call (ToolCallRecord): The ToolCallRecord object to save.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("save_tool_call is not implemented")


class LogToolCallStorage(ToolCallStorage):
    """ToolCallStorage that logs calls rather than persisting them.

    Useful for storages that don't care about tool_calls.
    """

    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Log the tool call.

        Args:
            tool_call (ToolCallRecord): The ToolCallRecord object to log.

        """
        logger().info(
            "Invoked {tool_name} with args: {tool_input}",
            tool_name=tool_call.tool_name,
            tool_input=tool_call.input,
        )
        logger().debug(
            f"Tool {tool_call.tool_name} executed in {tool_call.latency_seconds:.2f} seconds",
        )
        match tool_call.status:
            case ToolCallStatus.SUCCESS:
                logger().info("Tool output: {output}", output=tool_call.output)
            case ToolCallStatus.FAILED:
                logger().error("Tool returned error {output}", output=tool_call.output)
            case ToolCallStatus.NEED_CLARIFICATION:
                logger().info("Tool returned clarifications {output}", output=tool_call.output)


class Storage(PlanStorage, WorkflowStorage, ToolCallStorage):
    """Combined base class for Plan Workflow + Tool storages."""


class InMemoryStorage(PlanStorage, WorkflowStorage, LogToolCallStorage):
    """Simple storage class that keeps plans + workflows in memory.

    Tool Calls are logged via the LogToolCallStorage.
    """

    plans: dict[PlanUUID, Plan]
    workflows: dict[WorkflowUUID, Workflow]

    def __init__(self) -> None:
        """Initialize Storage."""
        self.plans = {}
        self.workflows = {}

    def save_plan(self, plan: Plan) -> None:
        """Add plan to dict.

        Args:
            plan (Plan): The Plan object to save.

        """
        self.plans[plan.id] = plan

    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Get plan from dict.

        Args:
            plan_id (PlanUUID): The UUID of the plan to retrieve.

        Returns:
            Plan: The Plan object associated with the provided plan_id.

        Raises:
            PlanNotFoundError: If the plan is not found.

        """
        if plan_id in self.plans:
            return self.plans[plan_id]
        raise PlanNotFoundError(plan_id)

    def save_workflow(self, workflow: Workflow) -> None:
        """Add workflow to dict.

        Args:
            workflow (Workflow): The Workflow object to save.

        """
        self.workflows[workflow.id] = workflow

    def get_workflow(self, workflow_id: WorkflowUUID) -> Workflow:
        """Get workflow from dict.

        Args:
            workflow_id (WorkflowUUID): The UUID of the workflow to retrieve.

        Returns:
            Workflow: The Workflow object associated with the provided workflow_id.

        Raises:
            WorkflowNotFoundError: If the workflow is not found.

        """
        if workflow_id in self.workflows:
            return self.workflows[workflow_id]
        raise WorkflowNotFoundError(workflow_id)

    def get_workflows(self, workflow_state: WorkflowState | None = None) -> list[Workflow]:
        """Get workflow from dict.

        Args:
            workflow_state (WorkflowState | None): Optionally filter workflows by their state.

        Returns:
            list[Workflow]: A list of Workflow objects that match the given state.

        """
        if not workflow_state:
            return list(self.workflows.values())
        return [
            workflow for workflow in self.workflows.values() if workflow.state == workflow_state
        ]


class DiskFileStorage(PlanStorage, WorkflowStorage, LogToolCallStorage):
    """Disk-based implementation of the Storage interface.

    Stores serialized Plan and Workflow objects as JSON files on disk.
    """

    def __init__(self, storage_dir: str | None) -> None:
        """Set storage dir.

        Args:
            storage_dir (str | None): Optional directory for storing files.

        """
        self.storage_dir = storage_dir or ".portia"

    def _ensure_storage(self) -> None:
        """Ensure that the storage directory exists.

        Raises:
            FileNotFoundError: If the directory cannot be created.

        """
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)

    def _write(self, file_name: str, content: BaseModel) -> None:
        """Write a serialized Plan or Workflow to a JSON file.

        Args:
            file_name (str): Name of the file to write.
            content (BaseModel): The Plan or Workflow object to serialize.

        """
        self._ensure_storage()  # Ensure storage directory exists
        with Path(self.storage_dir, file_name).open("w") as file:
            file.write(content.model_dump_json(indent=4))

    def _read(self, file_name: str, model: type[T]) -> T:
        """Read a JSON file and deserialize it into a BaseModel instance.

        Args:
            file_name (str): Name of the file to read.
            model (type[T]): The model class to deserialize into.

        Returns:
            T: The deserialized model instance.

        Raises:
            FileNotFoundError: If the file is not found.
            ValidationError: If the deserialization fails.

        """
        with Path(self.storage_dir, file_name).open("r") as file:
            f = file.read()
            return model.model_validate_json(f)

    def save_plan(self, plan: Plan) -> None:
        """Save a Plan object to the storage.

        Args:
            plan (Plan): The Plan object to save.

        """
        self._write(f"{plan.id}.json", plan)

    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Retrieve a Plan object by its ID.

        Args:
            plan_id (PlanUUID): The ID of the Plan to retrieve.

        Returns:
            Plan: The retrieved Plan object.

        Raises:
            PlanNotFoundError: If the Plan is not found or validation fails.

        """
        try:
            print(f"{plan_id}.json")
            return self._read(f"{plan_id}.json", Plan)
        except (ValidationError, FileNotFoundError) as e:
            raise PlanNotFoundError(plan_id) from e

    def save_workflow(self, workflow: Workflow) -> None:
        """Save a Workflow object to the storage.

        Args:
            workflow (Workflow): The Workflow object to save.

        """
        self._write(f"{workflow.id}.json", workflow)

    def get_workflow(self, workflow_id: WorkflowUUID) -> Workflow:
        """Retrieve a Workflow object by its ID.

        Args:
            workflow_id (WorkflowUUID): The ID of the Workflow to retrieve.

        Returns:
            Workflow: The retrieved Workflow object.

        Raises:
            WorkflowNotFoundError: If the Workflow is not found or validation fails.

        """
        try:
            return self._read(f"{workflow_id}.json", Workflow)
        except (ValidationError, FileNotFoundError) as e:
            raise WorkflowNotFoundError(workflow_id) from e

    def get_workflows(self, workflow_state: WorkflowState | None = None) -> list[Workflow]:
        """Find all workflows in storage that match state.

        Args:
            workflow_state (WorkflowState | None): Optionally filter workflows by their state.

        Returns:
            list[Workflow]: A list of Workflow objects that match the given state.

        """
        self._ensure_storage()

        workflows = []

        directory_path = Path(self.storage_dir)
        for f in directory_path.iterdir():
            if f.is_file() and f.name.startswith(WORKFLOW_UUID_PREFIX):
                workflow = self._read(f.name, Workflow)
                if not workflow_state or workflow.state == workflow_state:
                    workflows.append(workflow)

        return workflows


class PortiaCloudStorage(Storage):
    """Save plans, workflows and tool calls to portia cloud."""

    def __init__(self, config: Config) -> None:
        """Initialize the PortiaCloudStorage instance.

        Args:
            config (Config): The configuration containing API details for Portia Cloud.

        """
        self.api_key = config.must_get_api_key("portia_api_key")
        self.api_endpoint = config.must_get("portia_api_endpoint", str)

    def check_response(self, response: httpx.Response) -> None:
        """Validate the response from Portia API.

        Args:
            response (httpx.Response): The response from the Portia API to check.

        Raises:
            StorageError: If the response from the Portia API indicates an error.

        """
        if not response.is_success:
            error_str = str(response.content)
            logger().error(f"Error from Portia Cloud: {error_str}")
            raise StorageError(error_str)

    def save_plan(self, plan: Plan) -> None:
        """Save a plan to Portia Cloud.

        Args:
            plan (Plan): The Plan object to save to the cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        print(f"POST: {self.api_endpoint}/api/v0/plans/")
        try:
            response = httpx.post(
                url=f"{self.api_endpoint}/api/v0/plans/",
                json={
                    "id": str(plan.id),
                    "query": plan.plan_context.query,
                    "tool_ids": plan.plan_context.tool_ids,
                    "steps": [step.model_dump(mode="json") for step in plan.steps],
                },
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)

    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Retrieve a plan from Portia Cloud.

        Args:
            plan_id (PlanUUID): The ID of the plan to retrieve.

        Returns:
            Plan: The Plan object retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails or the plan does not exist.

        """
        print(f"GET: {self.api_endpoint}/api/v0/plans/{plan_id}/")
        try:
            response = httpx.get(
                url=f"{self.api_endpoint}/api/v0/plans/{plan_id}/",
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            response_json = response.json()
            return Plan(
                id=PlanUUID.from_string(response_json["id"]),
                plan_context=PlanContext(
                    query=response_json["query"],
                    tool_ids=response_json["tool_ids"],
                ),
                steps=[Step.model_validate(step) for step in response_json["steps"]],
            )

    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow to Portia Cloud.

        Args:
            workflow (Workflow): The Workflow object to save to the cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        print(f"POST: {self.api_endpoint}/api/v0/workflows/, id was {workflow.id!s}")
        try:
            response = httpx.post(
                url=f"{self.api_endpoint}/api/v0/workflows/",
                json={
                    "id": str(workflow.id),
                    "current_step_index": workflow.current_step_index,
                    "state": workflow.state,
                    "execution_context": workflow.execution_context.model_dump(mode="json"),
                    "outputs": workflow.outputs.model_dump(mode="json"),
                    "plan_id": str(workflow.plan_id),
                },
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            # If the workflow exists, update it instead
            if "workflow with this id already exists." in str(response.content):
                print(f"PATCH: {self.api_endpoint}/api/v0/workflows/{workflow.id}/")
                response = httpx.patch(
                    url=f"{self.api_endpoint}/api/v0/workflows/{workflow.id}/",
                    json={
                        "current_step_index": workflow.current_step_index,
                        "state": workflow.state,
                        "execution_context": workflow.execution_context.model_dump(mode="json"),
                        "outputs": workflow.outputs.model_dump(mode="json"),
                        "plan_id": str(workflow.plan_id),
                    },
                    headers={
                        "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                        "Content-Type": "application/json",
                    },
                    timeout=10,
                )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)

    def get_workflow(self, workflow_id: WorkflowUUID) -> Workflow:
        """Retrieve a workflow from Portia Cloud.

        Args:
            workflow_id (WorkflowUUID): The ID of the workflow to retrieve.

        Returns:
            Workflow: The Workflow object retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails or the workflow does not exist.

        """
        print(f"GET: {self.api_endpoint}/api/v0/workflows/{workflow_id}/")
        try:
            response = httpx.get(
                url=f"{self.api_endpoint}/api/v0/workflows/{workflow_id}/",
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            response_json = response.json()
            print(f"RESPONSE: {response_json}")
            return Workflow(
                id=WorkflowUUID.from_string(response_json["id"]),
                plan_id=PlanUUID.from_string(response_json["plan"]["id"]),
                current_step_index=response_json["current_step_index"],
                state=WorkflowState(response_json["state"]),
                execution_context=ExecutionContext.model_validate(
                    response_json["execution_context"],
                ),
                outputs=WorkflowOutputs.model_validate(response_json["outputs"]),
            )

    def get_workflows(self, workflow_state: WorkflowState | None = None) -> list[Workflow]:
        """Retrieve workflows from Portia Cloud.

        Args:
            workflow_state (WorkflowState | None): Optionally filter workflows by their state.

        Returns:
            list[Workflow]: A list of Workflow objects retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        print(f"GET: {self.api_endpoint}/api/v0/workflows/")
        try:
            query = ""
            if workflow_state:
                query = f"?workflow_state={workflow_state.value}"
            response = httpx.get(
                url=f"{self.api_endpoint}/api/v0/workflows/{query}",
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            response_json = response.json()
            return [
                Workflow(
                    id=WorkflowUUID.from_string(workflow["id"]),
                    plan_id=PlanUUID.from_string(workflow["plan"]["id"]),
                    current_step_index=workflow["current_step_index"],
                    state=WorkflowState(workflow["state"]),
                    execution_context=ExecutionContext.model_validate(
                        workflow["execution_context"],
                    ),
                    outputs=WorkflowOutputs.model_validate(workflow["outputs"]),
                )
                for workflow in response_json
            ]

    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Save a tool call to Portia Cloud.

        Args:
            tool_call (ToolCallRecord): The ToolCallRecord object to save to the cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        print(f"POST: {self.api_endpoint}/api/v0/tool-calls/")
        try:
            response = httpx.post(
                url=f"{self.api_endpoint}/api/v0/tool-calls/",
                json={
                    "workflow_id": str(tool_call.workflow_id),
                    "tool_name": tool_call.tool_name,
                    "step": tool_call.step,
                    "end_user_id": tool_call.end_user_id or "",
                    "additional_data": tool_call.additional_data,
                    "input": tool_call.input,
                    "output": tool_call.output,
                    "status": tool_call.status,
                    "latency_seconds": tool_call.latency_seconds,
                },
                headers={
                    "Authorization": f"Api-Key {self.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
    
    def _get_workflow_from_response(self, json_workflow: dict) -> Workflow:
        """Get a workflow from the response.

        Args:
            json_workflow (dict): The JSON response from the Portia Cloud API.

        Returns:
            Workflow: The Workflow object retrieved from the response.

        """
        if "outputs" in json_workflow:
            outputs = WorkflowOutputs.model_validate(json_workflow["outputs"])
        else:
            outputs = None
        return Workflow(
            id=WorkflowUUID.from_string(json_workflow["id"]),
            plan_id=PlanUUID.from_string(json_workflow["plan"]["id"]),
            current_step_index=json_workflow["current_step_index"],
            state=WorkflowState(json_workflow["state"]),
            execution_context=ExecutionContext.model_validate(json_workflow["execution_context"]),
            outputs=WorkflowOutputs.model_validate(json_workflow["outputs"]),
        )
