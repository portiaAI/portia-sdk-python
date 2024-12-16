"""Runner classes which actually plan + run queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.config import Config, InvalidStorageError, StorageClass
from portia.plan import Output, Plan
from portia.planner import PlanError, Planner
from portia.storage import DiskFileStorage, InMemoryStorage
from portia.tool_registry import LocalToolRegistry, ToolRegistry, ToolSet
from portia.workflow import InvalidWorkflowStateError, Workflow, WorkflowState

if TYPE_CHECKING:
    from portia.config import Config


class Runner:
    """Create and run plans for queries."""

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize storage and tools."""
        self.config = config
        self.tool_registry = tool_registry or LocalToolRegistry()

        match config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=config.must_get("storage_dir", str))
            case _:
                raise InvalidStorageError

    def run_query(
        self,
        query: str,
        tools: ToolSet | None = None,
        example_workflows: list[Plan] | None = None,
    ) -> Workflow:
        """Plan and run a query in one go."""
        plan = self.plan_query(query, tools, example_workflows)
        return self.run_plan(plan)

    def plan_query(
        self,
        query: str,
        tools: ToolSet | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        """Plans how to do the query given the set of tools and any examples."""
        if not tools:
            tools = self.tool_registry.match_tools(query)

        planner = Planner(config=self.config)
        outcome = planner.generate_plan_or_error(
            query=query,
            tool_list=tools,
            system_context=self.config.planner_system_content,
            examples=example_plans,
        )
        if outcome.error:
            raise PlanError(outcome.error)
        self.storage.save_plan(outcome.plan)
        return outcome.plan

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
