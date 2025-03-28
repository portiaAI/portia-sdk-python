"""Helpers to testing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, override

from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, SecretStr

from portia.clarification import Clarification, InputClarification
from portia.clarification_handler import ClarificationHandler
from portia.config import Config, LogLevel, StorageClass
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_context import ExecutionContext, empty_context
from portia.llm_wrapper import LLMWrapper
from portia.model import LangChainModel
from portia.plan import Plan, PlanContext, Step, Variable
from portia.plan_run import PlanRun, PlanRunUUID
from portia.tool import Tool, ToolRunContext
from portia.tool_call import ToolCallRecord, ToolCallStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from unittest.mock import MagicMock

    from mcp import ClientSession

    from portia.execution_context import ExecutionContext
    from portia.mcp_session import McpClientConfig


def get_test_tool_context(
    plan_run_id: PlanRunUUID | None = None,
    config: Config | None = None,
) -> ToolRunContext:
    """Return a test tool context."""
    if not plan_run_id:
        plan_run_id = PlanRunUUID()
    if not config:
        config = get_test_config()
    return ToolRunContext(
        execution_context=get_execution_ctx(),
        plan_run_id=plan_run_id,
        config=config,
        clarifications=[],
    )


def get_test_plan_run() -> tuple[Plan, PlanRun]:
    """Generate a simple test plan_run."""
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
    return plan, PlanRun(plan_id=plan.id, current_step_index=0)


def get_test_tool_call(plan_run: PlanRun) -> ToolCallRecord:
    """Return a test tool call record."""
    return ToolCallRecord(
        tool_name="",
        plan_run_id=plan_run.id,
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


def get_test_llm_wrapper(mock_invoker: MockInvoker) -> LLMWrapper:
    """Get a test LLM wrapper."""
    return LLMWrapper(model=LangChainModel(client=mock_invoker))


def get_execution_ctx(plan_run: PlanRun | None = None) -> ExecutionContext:
    """Return an execution context from a PlanRun."""
    if plan_run:
        return plan_run.execution_context
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
                plan_run_id=ctx.plan_run_id,
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


class TestClarificationHandler(ClarificationHandler):  # noqa: D101
    received_clarification: Clarification | None = None
    clarification_response: object = "Test"

    @override
    def handle_input_clarification(
        self,
        clarification: InputClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],
    ) -> None:
        self.received_clarification = clarification
        return on_resolution(clarification, self.clarification_response)

    def reset(self) -> None:
        """Reset the received clarification."""
        self.received_clarification = None


class MockMcpSessionWrapper:
    """Wrapper for mocking out an MCP ClientSession for testing MCP integration."""

    def __init__(self, session: MagicMock) -> None:
        """Initialize the wrapper."""
        self.session = session

    @asynccontextmanager
    async def mock_mcp_session(self, _: McpClientConfig) -> AsyncIterator[ClientSession]:
        """Mock method to swap out with the mcp_session context manager."""
        yield self.session


class MockInvoker:
    """Mock invoker."""

    called: bool
    prompt: ChatPromptValue | None
    response: AIMessage | BaseModel | None
    output_format: Any | None
    tools: Any | None
    method: str | None

    def __init__(self, response: AIMessage | BaseModel | None = None) -> None:
        """Init worker."""
        self.called = False
        self.prompt = None
        self.response = response
        self.output_format = None
        self.tools = None
        self.method = None

    def invoke(
        self,
        prompt: ChatPromptValue,
        _: RunnableConfig | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> AIMessage | BaseModel:
        """Mock run for invoking the chain."""
        self.called = True
        self.prompt = prompt
        if self.response:
            return self.response
        return AIMessage(content="invoked")

    def with_structured_output(
        self,
        output_format: Any,  # noqa: ANN401
        method: str = "function_calling",
    ) -> MockInvoker:
        """Model wrapper for structured output."""
        self.output_format = output_format
        self.method = method
        return self
