"""Tests for pull auth forward changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, HttpUrl

from portia.clarification import ActionClarification, Clarification, InputClarification
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.output import LocalOutput, Output
from portia.plan import PlanBuilder, Step
from portia.plan_run import PlanRun, PlanRunState
from portia.portia import Portia
from portia.tool import ReadyResponse, Tool, ToolRunContext, _ArgsSchemaPlaceholder
from tests.utils import get_test_config

if TYPE_CHECKING:
    from portia.prefixed_uuid import PlanRunUUID


class ReadyTool(Tool):
    """A tool that can be set to ready or not ready."""

    id: str = "ready_tool"
    name: str = "Ready Tool"
    description: str = "A tool that can be set to ready or not ready."
    args_schema: type[BaseModel] = _ArgsSchemaPlaceholder
    output_schema: tuple[str, str] = ("ReadyResponse", "A response from the tool")

    is_ready: bool | list[bool] = False
    auth_url: str = "https://fake.portiaai.test/auth"

    def _get_clarifications(self, plan_run_id: PlanRunUUID) -> list[Clarification]:
        """Generate clarifications for the ready check."""
        is_ready = (
            self.is_ready
            if isinstance(self.is_ready, bool)
            else self.is_ready.pop(0)
            if isinstance(self.is_ready, list) and len(self.is_ready) > 0
            else False
        )
        if is_ready:
            return []
        return [  # pyright: ignore[reportReturnType]
            ActionClarification(
                user_guidance="user guidance",
                plan_run_id=plan_run_id,
                action_url=HttpUrl(self.auth_url),
            )
        ]

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:
        """Is the tool ready."""
        clarifications = self._get_clarifications(ctx.plan_run_id)
        return ReadyResponse(
            ready=len(clarifications) == 0,
            clarifications=clarifications,
        )

    def run(self, ctx: ToolRunContext) -> None:  # noqa: ARG002
        """Run the tool."""
        return


def test_portia_resume_tool_requires_clarification() -> None:
    """Test that a tool that requires clarification gets raised at start of plan run."""
    ready_tool = ReadyTool()
    portia = Portia(config=get_test_config(), tools=[ready_tool])
    plan = PlanBuilder().step("", ready_tool.id).build()
    plan_run = portia.create_plan_run(plan, end_user="123")
    portia.storage.save_plan(plan)  # Explicitly save plan for test

    output_plan_run = portia.resume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 1
    outstanding_clarification = output_plan_run.get_outstanding_clarifications()[0]
    assert isinstance(outstanding_clarification, ActionClarification)
    assert outstanding_clarification.resolved is False
    assert outstanding_clarification.plan_run_id == plan_run.id
    assert str(outstanding_clarification.action_url) == "https://fake.portiaai.test/auth"


def test_portia_resume_multiple_instances_of_same_tool() -> None:
    """Test clarification handling for multiple instances of the same tool.

    Only one clarification should be raised for the tool.
    """
    ready_tool = ReadyTool()
    portia = Portia(config=get_test_config(), tools=[ready_tool, ready_tool])
    plan = PlanBuilder().step("1", ready_tool.id).step("2", ready_tool.id).build()
    plan_run = portia.create_plan_run(plan, end_user="123")
    portia.storage.save_plan(plan)  # Explicitly save plan for test

    output_plan_run = portia.resume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 1
    outstanding_clarification = output_plan_run.get_outstanding_clarifications()[0]
    assert isinstance(outstanding_clarification, ActionClarification)
    assert outstanding_clarification.resolved is False
    assert outstanding_clarification.plan_run_id == plan_run.id
    assert str(outstanding_clarification.action_url) == "https://fake.portiaai.test/auth"


def test_portia_resume_multiple_tools_require_clarification() -> None:
    """Test clarifications are raised for multiple tools in a plan run if they require it."""
    ready_tool = ReadyTool(auth_url="https://fake.portiaai.test/auth")
    ready_tool_2 = ReadyTool(id="ready_tool_2", auth_url="https://fake.portiaai.test/auth2")
    portia = Portia(config=get_test_config(), tools=[ready_tool, ready_tool_2])
    plan = PlanBuilder().step("1", ready_tool.id).step("2", ready_tool_2.id).build()
    plan_run = portia.create_plan_run(plan, end_user="123")
    portia.storage.save_plan(plan)  # Explicitly save plan for test

    output_plan_run = portia.resume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 2
    outstanding_clarifications = output_plan_run.get_outstanding_clarifications()
    assert isinstance(outstanding_clarifications[0], ActionClarification)
    assert outstanding_clarifications[0].plan_run_id == plan_run.id
    assert str(outstanding_clarifications[0].action_url) == "https://fake.portiaai.test/auth"
    assert outstanding_clarifications[0].resolved is False
    assert isinstance(outstanding_clarifications[1], ActionClarification)
    assert outstanding_clarifications[1].plan_run_id == plan_run.id
    assert str(outstanding_clarifications[1].action_url) == "https://fake.portiaai.test/auth2"
    assert outstanding_clarifications[1].resolved is False


class RaiseClarificationAgent(BaseExecutionAgent):
    """A dummy execution agent that raises a clarification on run."""

    def execute_sync(self) -> Output:
        """Execute the agent - return a clarification."""
        return LocalOutput(
            value=[
                InputClarification(
                    user_guidance="user guidance",
                    plan_run_id=self.plan_run.id,
                    argument_name="argument_name",
                )
            ]
        )


class CustomPortia(Portia):
    """A custom portia that uses a custom execution agent."""

    def _get_agent_for_step(self, step: Step, plan_run: PlanRun) -> BaseExecutionAgent:
        if step.task == "raise_clarification":
            tool = self._get_tool_for_step(step, plan_run)
            return RaiseClarificationAgent(
                step=step,
                plan_run=plan_run,
                config=self.config,
                end_user=self.initialize_end_user(plan_run.end_user_id),
                agent_memory=self.storage,
                tool=tool,
            )
        return super()._get_agent_for_step(step, plan_run)


def test_tool_raise_clarification_all_remaining_tool_ready_status_rechecked() -> None:
    """Test that all remaining steps have their tool ready status checked on any interruption."""
    ready_tool = ReadyTool(is_ready=True)
    ready_once_tool = ReadyTool(id="ready_once_tool", is_ready=[True, False])
    portia = CustomPortia(config=get_test_config(), tools=[ready_tool, ready_once_tool])
    plan = (
        PlanBuilder()
        .step("raise_clarification", ready_tool.id)
        .step("2", ready_once_tool.id)
        .build()
    )
    plan_run = portia.create_plan_run(plan, end_user="123")
    portia.storage.save_plan(plan)  # Explicitly save plan for test

    output_plan_run = portia.resume(plan_run)
    assert output_plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert len(output_plan_run.get_outstanding_clarifications()) == 2
    outstanding_clarifications = output_plan_run.get_outstanding_clarifications()
    assert isinstance(outstanding_clarifications[0], InputClarification)
    assert outstanding_clarifications[0].plan_run_id == plan_run.id
    assert outstanding_clarifications[0].argument_name == "argument_name"
    assert outstanding_clarifications[0].resolved is False
    assert isinstance(outstanding_clarifications[1], ActionClarification)
    assert outstanding_clarifications[1].plan_run_id == plan_run.id
    assert str(outstanding_clarifications[1].action_url) == ready_tool.auth_url
    assert outstanding_clarifications[1].resolved is False
