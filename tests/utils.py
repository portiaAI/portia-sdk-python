"""Helpers to testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, SecretStr

from portia.clarification import Clarification, InputClarification
from portia.config import Config, LogLevel, StorageClass
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_context import ExecutionContext, empty_context
from portia.plan import Plan, PlanContext, Step, Variable
from portia.tool import Tool, ToolRunContext
from portia.tool_call import ToolCallRecord, ToolCallStatus
from portia.workflow import Workflow, WorkflowUUID

if TYPE_CHECKING:
    from portia.execution_context import ExecutionContext


def get_test_tool_context(
    workflow_id: WorkflowUUID | None = None,
    config: Config | None = None,
) -> ToolRunContext:
    """Return a test tool context."""
    if not workflow_id:
        workflow_id = WorkflowUUID()
    if not config:
        config = get_test_config()
    return ToolRunContext(
        execution_context=get_execution_ctx(),
        workflow_id=workflow_id,
        config=config,
        clarifications=[],
    )


def get_test_workflow() -> tuple[Plan, Workflow]:
    """Generate a simple test workflow."""
    step1 = Step(
        task="Add 1 + 2",
        inputs=[
            Variable(name="a", value=1, description=""),
            Variable(name="b", value=2, description=""),
        ],
        output="$sum",
    )
    plan = Plan(
        plan_context=PlanContext(
            query="Add 1 + 2",
            tool_ids=["add_tool"],
        ),
        steps=[step1],
    )
    return plan, Workflow(plan_id=plan.id, current_step_index=0)


def get_test_tool_call(workflow: Workflow) -> ToolCallRecord:
    """Return a test tool call record."""
    return ToolCallRecord(
        tool_name="",
        workflow_id=workflow.id,
        step=1,
        end_user_id="1",
        additional_data={},
        output={},
        input={},
        latency_seconds=10,
        status=ToolCallStatus.SUCCESS,
    )


def get_test_config(**kwargs) -> Config:  # noqa: ANN003
    """Get test config."""
    return Config.from_default(
        **kwargs,
        default_log_level=LogLevel.INFO,
        openai_api_key=SecretStr("123"),
        storage_class=StorageClass.MEMORY,
    )


def get_execution_ctx(workflow: Workflow | None = None) -> ExecutionContext:
    """Return an execution context from a workflow."""
    if workflow:
        return workflow.execution_context
    return empty_context()


class AdditionToolSchema(BaseModel):
    """Input for AdditionTool."""

    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")


class AdditionTool(Tool):
    """Adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Takes two numbers and adds them together"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("int", "int: The value of the addition")

    def run(self, _: ToolRunContext, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b


class ClarificationToolSchema(BaseModel):
    """Input for ClarificationTool."""

    user_guidance: str


class ClarificationTool(Tool):
    """Returns a Clarification."""

    id: str = "clarification_tool"
    name: str = "Clarification Tool"
    description: str = "Returns a clarification"
    args_schema: type[BaseModel] = ClarificationToolSchema
    output_schema: tuple[str, str] = (
        "Clarification",
        "Clarification: The value of the Clarification",
    )

    def run(
        self,
        ctx: ToolRunContext,
        user_guidance: str,
    ) -> Clarification | None:
        """Add the numbers."""
        if len(ctx.clarifications) == 0:
            return InputClarification(
                workflow_id=ctx.workflow_id,
                user_guidance=user_guidance,
                argument_name="raise_clarification",
            )
        return None


class MockToolSchema(BaseModel):
    """Input for MockTool."""


class MockTool(Tool):
    """A mock tool class for testing purposes."""

    name: str = "Mock Tool"
    description: str = "do nothing"
    args_schema: type[BaseModel] = MockToolSchema
    output_schema: tuple[str, str] = ("None", "None: returns nothing")

    def run(
        self,
        _: ToolRunContext,
    ) -> None:
        """Do nothing."""
        return


class ErrorToolSchema(BaseModel):
    """Input for ErrorTool."""

    error_str: str
    return_soft_error: bool
    return_uncaught_error: bool


class ErrorTool(Tool):
    """Returns an Error."""

    id: str = "error_tool"
    name: str = "Error Tool"
    description: str = "Returns a error"
    args_schema: type[BaseModel] = ErrorToolSchema
    output_schema: tuple[str, str] = (
        "Error",
        "Error: The value of the error",
    )

    def run(
        self,
        _: ToolRunContext,
        error_str: str,
        return_uncaught_error: bool,  # noqa: FBT001
        return_soft_error: bool,  # noqa: FBT001
    ) -> None:
        """Return the error."""
        if return_uncaught_error:
            raise Exception(error_str)  # noqa: TRY002
        if return_soft_error:
            raise ToolSoftError(error_str)
        raise ToolHardError(error_str)


class NoneTool(Tool):
    """Returns None."""

    id: str = "none_tool"
    name: str = "None Tool"
    description: str = "returns None"
    output_schema: tuple[str, str] = ("None", "None: nothing")

    def run(self, _: ToolRunContext) -> None:
        """Return."""
        return
