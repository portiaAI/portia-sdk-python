"""Helpers to testing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, override
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, SecretStr

from portia.clarification import Clarification, InputClarification
from portia.clarification_handler import ClarificationHandler
from portia.config import Config, LogLevel, StorageClass
from portia.end_user import EndUser
from portia.errors import ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalDataValue
from portia.model import GenerativeModel, LangChainGenerativeModel
from portia.plan import Plan, PlanContext, Step, Variable
from portia.plan_run import PlanRun, PlanRunUUID
from portia.tool import Tool, ToolRunContext
from portia.tool_call import ToolCallRecord, ToolCallStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from mcp import ClientSession

    from portia.mcp_session import McpClientConfig


def get_test_tool_context(
    plan_run_id: PlanRunUUID | None = None,
    config: Config | None = None,
    end_user: EndUser | None = None,
) -> ToolRunContext:
    """Return a test tool context."""
    if not plan_run_id:
        plan_run_id = PlanRunUUID()
    if not config:
        config = get_test_config()
    if not end_user:
        end_user = EndUser(external_id="test")

    return ToolRunContext(
        plan_run_id=plan_run_id,
        config=config,
        clarifications=[],
        end_user=end_user,
    )


def get_test_plan_run() -> tuple[Plan, PlanRun]:
    """Generate a simple test plan_run."""
    step1 = Step(
        task="Add $a + 2",
        inputs=[
            Variable(name="$a", description="the first number"),
        ],
        output="$sum",
    )
    plan = Plan(
        plan_context=PlanContext(
            query="Add $a + 2",
            tool_ids=["add_tool"],
        ),
        steps=[step1],
    )
    plan_run = PlanRun(plan_id=plan.id, current_step_index=0, end_user_id="test")
    plan_run.outputs.step_outputs = {
        "$a": LocalDataValue(value="3"),
    }
    return plan, plan_run


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


class AdditionToolSchema(BaseModel):
    """Input for AdditionTool."""

    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")


class AdditionTool(Tool):
    """Adds two numbers."""

    id: str = "add_tool"
    name: str = "Add Tool"
    description: str = "Use this tool to add two numbers together, it takes two numbers a + b"
    args_schema: type[BaseModel] = AdditionToolSchema
    output_schema: tuple[str, str] = ("int", "int: The value of the addition")

    def run(self, _: ToolRunContext, a: int, b: int) -> int:
        """Add the numbers."""
        return a + b


class EndUserUpdateToolSchema(BaseModel):
    """Input for AdditionTool."""

    name: str | None = Field(default=None, description="The new name for the end user.")


class EndUserUpdateTool(Tool):
    """Adds two numbers."""

    id: str = "end_user_update"
    name: str = "End User Update Tool"
    description: str = "Updates the name of the end user"
    args_schema: type[BaseModel] = EndUserUpdateToolSchema
    output_schema: tuple[str, str] = ("str", "str: The new name")

    def run(self, ctx: ToolRunContext, name: str) -> str:
        """Change the name."""
        ctx.end_user.name = name
        return name


class ClarificationToolSchema(BaseModel):
    """Input for ClarificationTool."""

    user_guidance: str = Field(..., description="The user guidance for the clarification")


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


def get_mock_base_chat_model(
    response: Any = None,  # noqa: ANN401
) -> MagicMock:
    """Get a mock base chat model."""
    model = MagicMock(spec=BaseChatModel)

    def invoke(*_: Any, **__: Any) -> BaseMessage:
        """Mock invoke."""
        assert response is not None
        return response

    def with_structured_output(_: BaseModel, *__: Any, **___: Any) -> BaseChatModel:
        """Mock with structured output."""
        return model

    def bind_tools(_: Sequence[BaseTool], *__: Any, **___: Any) -> BaseChatModel:
        """Mock bind tools."""
        return model

    model.invoke.side_effect = invoke
    model.with_structured_output.side_effect = with_structured_output
    model.bind_tools.side_effect = bind_tools
    return model


def get_mock_generative_model(response: Any = None) -> GenerativeModel:  # noqa: ANN401
    """Get a mock generative model."""
    return LangChainGenerativeModel(
        client=get_mock_base_chat_model(response),
        model_name="test",
    )


def assert_clarification_equality_without_uuid(
    clarification1: Clarification,
    clarification2: Clarification,
) -> None:
    """Assert that two clarifications are equal without comparing the UUID.

    Args:
        clarification1: First clarification to compare
        clarification2: Second clarification to compare

    Raises:
        AssertionError: If the clarifications differ in any field except UUID

    """
    # Get dictionaries of both objects
    dict1 = clarification1.model_dump()
    dict2 = clarification2.model_dump()

    # Remove UUID fields from comparison
    dict1.pop("id", None)
    dict2.pop("id", None)

    # Assert all remaining fields are equal
    assert dict1 == dict2, f"Clarifications differ in fields other than UUID:\n{dict1}\nvs\n{dict2}"
