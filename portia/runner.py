"""Runner classes which actually plan + run queries."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from portia.plan import Output, Plan, Step, Variable
from portia.storage import DiskFileStorage, InMemoryStorage
from portia.tool_registry import LocalToolRegistry, ToolRegistry, ToolSet
from portia.workflow import InvalidWorkflowStateError, Workflow, WorkflowState

if TYPE_CHECKING:
    from portia.tool import Tool


class InvalidStorageError(Exception):
    """Raised when an invalid storage is provided."""


class StorageClass(Enum):
    """Represent locations plans and workflows are written to."""

    MEMORY = "MEMORY"
    DISK = "DISK"
    CLOUD = "CLOUD"


class RunnerConfig(BaseModel):
    """General configuration for the library."""

    portia_api_key: str | None = os.getenv("PORTIA_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    storage_class: StorageClass = StorageClass.MEMORY

    @classmethod
    def from_file(cls, file_path: Path) -> RunnerConfig:
        """Load configuration from a JSON file."""
        with Path.open(file_path) as f:
            return cls.model_validate_json(f.read())


class Runner:
    """Create and run plans for queries."""

    def __init__(
        self,
        config: RunnerConfig,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize storage and tools."""
        match config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(None)
            case _:
                raise InvalidStorageError

        self.tool_registry = tool_registry or LocalToolRegistry()

    def run_query(
        self,
        query: str,
        tools: list[Tool] | None = None,
        example_workflows: list[Plan] | None = None,
    ) -> Workflow:
        """Plan and run a query in one go."""
        plan = self.plan_query(query, tools, example_workflows)
        return self.run_plan(plan)

    def plan_query(
        self,
        query: str,
        tools: list[Tool] | None = None,
        example_plans: list[Plan] | None = None,  # noqa: ARG002 - we're not using example plans yet
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples."""
        tool_set = ToolSet(tools) if tools else self.tool_registry.match_tools(query)
        steps = [
            Step(
                tool_name=tool.name,
                task="Do something",
                input=[
                    Variable(name="a", value=4, description="A value"),
                    Variable(name="b", value=5, description="B value"),
                ],
                output="$sum",
            )
            for tool in tool_set.get_tools()
        ]
        plan = Plan(query=query, steps=steps)
        self.storage.save_plan(plan)
        return plan

    def run_plan(self, plan: Plan) -> Workflow:
        """Run a plan returning the completed workflow or clarifications if needed."""
        workflow = Workflow(plan_id=plan.id, state=WorkflowState.IN_PROGRESS)
        return self._execute_workflow(plan, workflow)

    def resume_workflow(self, workflow: Workflow) -> Workflow:
        """Resume a workflow after an interruption."""
        if workflow.state not in [WorkflowState.IN_PROGRESS, WorkflowState.NEED_CLARIFICATION]:
            raise InvalidWorkflowStateError
        plan = self.storage.get_plan(plan_id=workflow.plan_id)
        return self._execute_workflow(plan, workflow)

    def _execute_workflow(self, plan: Plan, workflow: Workflow) -> Workflow:
        self.storage.save_workflow(workflow)
        for index in range(workflow.current_step_index, len(plan.steps)):
            step = plan.steps[index]
            workflow.current_step_index = index
            if step.tool_name:
                try:
                    tool = self.tool_registry.get_tool(step.tool_name)
                    args = {var.name: var.value for var in step.input} if step.input else {}
                    output = tool.run(**args)
                except Exception as e:  # noqa: BLE001
                    workflow.step_outputs[step.output] = Output(value=str(e))
                    workflow.state = WorkflowState.FAILED
                    self.storage.save_workflow(workflow)
                    return workflow
                else:
                    workflow.step_outputs[step.output] = Output(value=output)
                self.storage.save_workflow(workflow)
        workflow.state = WorkflowState.COMPLETE
        self.storage.save_workflow(workflow)
        return workflow
