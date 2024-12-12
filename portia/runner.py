""""""

from __future__ import annotations

from re import S
from typing import Sequence
from portia.plan import Output, Plan, Step, Variable, Workflow, WorkflowState
from portia.plan_storage import DiskFileStorage, InMemoryStorage, Storage

from portia.tool import Tool
from portia.tool_registry import LocalToolRegistry, ToolRegistry, ToolSet


class Runner:
    """"""

    def __init__(
        self,
        tools: Sequence[Tool] | None = None,
        api_key: str | None = None,
        storage: Storage | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.storage = storage or InMemoryStorage()
        self.tool_registry = tool_registry or LocalToolRegistry()

        # if api_key:
        #     storage += PortiaWorkflowStorage(api_key)
        #     registry += PortiaToolRegistry(api_key)

        if tools:
            self.tool_registry += LocalToolRegistry.from_local_tools(tools)

    def run_query(
        self,
        query: str,
        tools: list[Tool] | None = None,
        example_workflows: list[Plan] | None = None,
    ) -> Workflow:
        plan = self.plan_query(query, tools, example_workflows)

        return self.run_plan(plan)

    def plan_query(
        self,
        query: str,
        tools: list[Tool] | None = None,
        example_plans: list[Plan] | None = None,
    ) -> Plan:
        tool_set = ToolSet(tools) if tools else self.tool_registry.match_tools(query)
        steps = [
            Step(
                tool_name=tool.name,
                task=tool.description,
                input=[
                    Variable(name="a", value=4, description="A value"),
                    Variable(name="b", value=5, description="B value"),
                ],
            )
            for tool in tool_set.tools
        ]
        plan = Plan(query=query, steps=steps)
        self.storage.save_plan(plan)
        return plan

    def run_plan(self, plan: Plan) -> Workflow:
        workflow = Workflow(plan=plan, state=WorkflowState.IN_PROGRESS)
        self.storage.save_workflow(workflow)
        for index, step in enumerate(workflow.plan.steps):
            workflow.current_step_index = index
            if step.tool_name:
                tool = self.tool_registry.get_tool(step.tool_name)
                try:
                    args = {var.name: var.value for var in step.input} if step.input else {}
                    output = tool.run(**args)
                except Exception as e:  # noqa: BLE001
                    workflow.step_outputs[step.id] = Output(value=str(e))
                    workflow.state = WorkflowState.FAILED
                    self.storage.save_workflow(workflow)
                    return workflow
                else:
                    workflow.step_outputs[step.id] = Output(value=output)
                self.storage.save_workflow(workflow)
        workflow.state = WorkflowState.COMPLETE
        self.storage.save_workflow(workflow)
        return workflow
