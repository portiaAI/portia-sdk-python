"""Tests for the Tool class."""

import json
from enum import Enum
from unittest.mock import MagicMock, patch

import httpx
import mcp
import pytest
from mcp import ClientSession
from pydantic import BaseModel, HttpUrl
from pytest_httpx import HTTPXMock

from portia.clarification import (
    ActionClarification,
    ClarificationCategory,
    ClarificationUUID,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalDataValue
from portia.mcp_session import StdioMcpClientConfig
from portia.tool import PortiaMcpTool, PortiaRemoteTool, Tool, ToolRunContext
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    ErrorTool,
    MockMcpSessionWrapper,
    get_test_config,
    get_test_tool_context,
)


@pytest.fixture
def add_tool() -> AdditionTool:
    """Fixture to create a mock tool instance."""
    return AdditionTool()


@pytest.fixture
def clarification_tool() -> ClarificationTool:
    """Fixture to create a mock tool instance."""
    return ClarificationTool()


def test_tool_initialization(add_tool: AdditionTool) -> None:
    """Test initialization of a Tool."""
    assert add_tool.name == "Add Tool"
    assert (
        add_tool.description
        == "Use this tool to add two numbers together, it takes two numbers a + b"
    )


def test_tool_initialization_long_description() -> None:
    """Test initialization of a Tool."""

    class FakeAdditionTool(AdditionTool):
        description: str = "this is a description" * 1000

    with pytest.raises(InvalidToolDescriptionError):
        FakeAdditionTool()


def test_run_signature_validation() -> None:
    """Check that invalid run signatures raise a validation error."""

    class TestArgSchema(BaseModel):
        foo: str

    class BadTool(Tool):
        id: str = "bad_tool"
        name: str = "Bad Tool"
        description: str = "bad"
        args_schema: type[BaseModel] = TestArgSchema
        output_schema: tuple[str, str] = ("str", "out")

        def run(self, ctx: ToolRunContext, foo: int) -> str:  # noqa: ARG002
            return "bad"

    # test the logs
    with patch("portia.tool.logger") as mock_logger:
        BadTool()
        mock_logger.return_value.warning.assert_called_once_with(
            "Run method argument 'foo' type <class 'int'> does not match "
            "args_schema field type: <class 'str'>"
        )


def test_run_signature_validation_complex_type() -> None:
    """Check that complex type mismatches raise a validation error."""

    class ComplexSchema(BaseModel):
        foo: dict[str, list[int]]

    class BadTool(Tool):
        id: str = "bad_tool_complex"
        name: str = "Bad Tool Complex"
        description: str = "bad"
        args_schema: type[BaseModel] = ComplexSchema
        output_schema: tuple[str, str] = ("str", "out")

        def run(self, ctx: ToolRunContext, foo: list[str]) -> str:  # noqa: ARG002
            return "bad"

    with patch("portia.tool.logger") as mock_logger:
        BadTool()
        mock_logger.return_value.warning.assert_called_once_with(
            "Run method argument 'foo' type list[str] does not match "
            "args_schema field type: dict[str, list[int]]"
        )


def test_run_signature_context_type_required() -> None:
    """Check that first argument must be annotated as ToolRunContext."""

    class TestArgSchema(BaseModel):
        foo: str

    class BadTool(Tool):
        id: str = "bad_tool_ctx"
        name: str = "Bad Tool Context"
        description: str = "bad"
        args_schema: type[BaseModel] = TestArgSchema
        output_schema: tuple[str, str] = ("str", "out")

        def run(self, ctx: int, foo: str) -> str:  # type: ignore[no-redef]  # noqa: ARG002
            return "bad"

    with patch("portia.tool.logger") as mock_logger:
        BadTool()
        mock_logger.return_value.warning.assert_called_once_with(
            "First argument of run must be annotated as ToolRunContext"
        )


def test_run_signature_validation_no_args() -> None:
    """Check that no args is valid."""

    class TestTool(Tool):
        id: str = "test_tool"
        name: str = "Test Tool"
        description: str = "test"
        output_schema: tuple[str, str] = ("str", "out")

        def run(self) -> str:
            return "test"

    with patch("portia.tool.logger") as mock_logger:
        TestTool()
        mock_logger.return_value.warning.assert_not_called()


def test_tool_to_langchain() -> None:
    """Test langchain rep of a Tool."""
    tool = AdditionTool()
    tool.to_langchain(ctx=get_test_tool_context())


def test_run_method(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    ctx = get_test_tool_context()
    result = add_tool.run(ctx, a, b)
    assert result == a + b


def test_handle(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    ctx = get_test_tool_context()
    result = add_tool.run(ctx, a, b)
    assert result == a + b


def test_run_method_with_uncaught_error() -> None:
    """Test the _run method wraps errors."""
    tool = ErrorTool()
    with pytest.raises(ToolSoftError):
        tool._run(
            ctx=get_test_tool_context(),
            error_str="this is an error",
            return_uncaught_error=True,
            return_soft_error=False,
        )


def test_ready() -> None:
    """Test the ready method."""
    tool = ErrorTool()
    assert tool.ready(get_test_tool_context()).ready


def test_tool_serialization() -> None:
    """Test tools can be serialized to string."""
    tool = AdditionTool()
    assert str(tool) == (
        f"ToolModel(id={tool.id!r}, name={tool.name!r}, "
        f"description={tool.description!r}, "
        f"args_schema={tool.args_schema.__name__!r}, "
        f"output_schema={tool.output_schema!r})"
    )
    # check we can also serialize to JSON
    AdditionTool().model_dump_json()


def test_remote_tool_run_with_pydantic_model(httpx_mock: HTTPXMock) -> None:
    """Test remote tool run with Pydantic model."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={"output": {"value": "Success"}},
    )

    class DoubleNestedSchema(BaseModel):
        d: int

    class NestedSchema(BaseModel):
        c: int
        double_nested: DoubleNestedSchema

    class TestArgSchema(BaseModel):
        a: int
        b: int
        sub: NestedSchema

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        args_schema=TestArgSchema,
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    res = tool.run(
        get_test_tool_context(),
        a=1,
        b=2,
        sub={"c": 3, "double_nested": DoubleNestedSchema(d=4)},
    )
    assert res == "Success"


def test_remote_tool_hard_error_from_server(httpx_mock: HTTPXMock) -> None:
    """Test http errors come back to hard errors."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        status_code=500,
        json={"output": {"value": "An error occurred."}},
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }

    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_remote_tool_run_with_unserializable_object() -> None:
    """Test remote tool run with unserializable object."""
    endpoint = "https://api.fake-portia.test"

    class UnserializableObject:
        pass

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    ctx = get_test_tool_context()
    with pytest.raises(
        ToolHardError,
        match="Object of type <class 'tests.unit.test_tool."
        "test_remote_tool_run_with_unserializable_object"
        ".<locals>.UnserializableObject'> is not JSON serializable",
    ):
        tool.run(ctx, some_object=UnserializableObject())


def test_remote_tool_soft_error(httpx_mock: HTTPXMock) -> None:
    """Test remote soft errors come back to soft errors."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={"output": {"value": "ToolSoftError: An error occurred."}, "soft_error": True},
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )

    ctx = get_test_tool_context()
    with pytest.raises(ToolSoftError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }
    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_remote_tool_bad_response(httpx_mock: HTTPXMock) -> None:
    """Test remote soft errors come back to soft errors."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={"ot": {"value": "An error occurred."}},
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )

    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }

    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_remote_tool_run_unhandled_error(httpx_mock: HTTPXMock) -> None:
    """Test tool ready unhandled error."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_exception(
        url=f"{endpoint}/api/v0/tools/test/run/",
        exception=httpx.HTTPError("Unhandled error"),
    )
    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    with pytest.raises(ToolHardError, match="Unhandled error"):
        tool.run(get_test_tool_context())


@pytest.mark.parametrize(
    ("response_json", "is_ready"),
    [
        ({"success": "true"}, True),
        ({}, False),
        ({"ready": True, "clarifications": []}, True),
        ({"ready": False, "clarifications": []}, False),
    ],
)
def test_remote_tool_ready(httpx_mock: HTTPXMock, response_json: dict, is_ready: bool) -> None:
    """Test remote tool ready."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/ready/",
        json=response_json,
    )
    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    ctx = get_test_tool_context()
    assert tool.ready(ctx).ready == is_ready

    content = {
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
        },
    }
    assert len(httpx_mock.get_requests()) == 1
    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/ready/",
            match_json=content,
        )
        is not None
    )


@pytest.mark.parametrize(
    ("status_code", "is_ready"),
    [(500, False), (404, False), (200, True)],
)
def test_remote_tool_ready_error(httpx_mock: HTTPXMock, status_code: int, is_ready: bool) -> None:
    """Test remote tool ready."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/ready/",
        status_code=status_code,
        json={"success": "true"},
    )
    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )

    ctx = get_test_tool_context()
    assert tool.ready(ctx).ready == is_ready


def test_remote_tool_action_clarifications(httpx_mock: HTTPXMock) -> None:
    """Test action clarifications."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={
            "output": {
                "value": [
                    {
                        "id": str(ClarificationUUID()),
                        "category": "Action",
                        "action_url": "https://example.com",
                        "user_guidance": "blah",
                    },
                ],
            },
        },
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, ActionClarification)
    assert output.action_url == HttpUrl("https://example.com")

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }

    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_remote_tool_input_clarifications(httpx_mock: HTTPXMock) -> None:
    """Test Input clarifications."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={
            "output": {
                "value": [
                    {
                        "id": str(ClarificationUUID()),
                        "category": "Input",
                        "user_guidance": "blah",
                        "argument_name": "t",
                    },
                ],
            },
        },
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, InputClarification)
    assert output.argument_name == "t"

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }
    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_remote_tool_mc_clarifications(httpx_mock: HTTPXMock) -> None:
    """Test Multi Choice clarifications."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={
            "output": {
                "value": [
                    {
                        "id": str(ClarificationUUID()),
                        "category": "Multiple Choice",
                        "user_guidance": "blah",
                        "argument_name": "t",
                        "options": [1],
                    },
                ],
            },
        },
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )
    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, MultipleChoiceClarification)
    assert output.options == [1]

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }
    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_remote_tool_value_confirm_clarifications(httpx_mock: HTTPXMock) -> None:
    """Test value confirm clarifications."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={
            "output": {
                "value": [
                    {
                        "id": str(ClarificationUUID()),
                        "category": "Value Confirmation",
                        "user_guidance": "blah",
                        "argument_name": "t",
                    },
                ],
            },
        },
    )

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=httpx.Client(base_url=endpoint),
    )

    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, ValueConfirmationClarification)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.end_user.external_id,
            "plan_run_id": str(ctx.plan_run.id),
            "additional_data": ctx.end_user.additional_data,
        },
    }

    assert (
        httpx_mock.get_request(
            method="POST",
            url=f"{endpoint}/api/v0/tools/test/run/",
            match_json=content,
        )
        is not None
    )


def test_portia_mcp_tool_call() -> None:
    """Test invoking a tool via MCP."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.call_tool.return_value = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text="Hello, world!")],
        isError=False,
    )

    class MyEnum(str, Enum):
        A = "A"

    class TestArgSchema(BaseModel):
        a: MyEnum
        b: int

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:test_tool",
        name="test_tool",
        description="I am a tool",
        output_schema=("str", "Tool output formatted as a JSON string"),
        args_schema=TestArgSchema,
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
        ),
    )
    expected = (
        '{"meta":null,"content":[{"type":"text","text":"Hello, world!","annotations":null,"meta":null}],'  # noqa: E501
        '"structuredContent":null,"isError":false}'
    )

    with patch(
        "portia.tool.get_mcp_session",
        new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
    ):
        tool_result = tool.run(get_test_tool_context(), a=1, b=2)
        assert tool_result == expected


def test_portia_mcp_tool_call_with_error() -> None:
    """Test invoking a tool via MCP."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.call_tool.return_value = mcp.types.CallToolResult(
        content=[],
        isError=True,
    )

    class TestArgSchema(BaseModel):
        a: int
        b: int

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:test_tool",
        name="test_tool",
        description="I am a tool",
        output_schema=("str", "Tool output formatted as a JSON string"),
        args_schema=TestArgSchema,
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
        ),
    )

    with (
        patch(
            "portia.tool.get_mcp_session",
            new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
        ),
        pytest.raises(ToolHardError),
    ):
        tool.run(get_test_tool_context(), a=1, b=2)


@pytest.mark.asyncio
async def test_portia_mcp_tool_call_with_timeout() -> None:
    """Test that the timeout takes effect."""
    mock_mcp_session = MockMcpSessionWrapper(MagicMock(spec=ClientSession))
    mock_mcp_session.session.call_tool.side_effect = mcp.McpError(
        mcp.types.ErrorData(
            code=httpx.codes.REQUEST_TIMEOUT,
            message="Request timed out",
        ),
    )

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:test_tool",
        name="test_tool",
        description="I am a tool",
        output_schema=("str", "Tool output formatted as a JSON string"),
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
            tool_call_timeout_seconds=1,
        ),
    )

    with (
        patch(
            "portia.tool.get_mcp_session",
            new=mock_mcp_session.mock_mcp_session,
        ),
        pytest.raises(ToolSoftError),
    ):
        await tool.arun(get_test_tool_context(), a=1, b=2)


# Async tests for PortiaMcpTool
@pytest.mark.asyncio
async def test_portia_mcp_tool_async_call() -> None:
    """Test invoking a tool via MCP asynchronously."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.call_tool.return_value = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text="Hello, world!")],
        isError=False,
    )

    class MyEnum(str, Enum):
        A = "A"

    class TestArgSchema(BaseModel):
        a: MyEnum
        b: int

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:test_tool",
        name="test_tool",
        description="I am a tool",
        output_schema=("str", "Tool output formatted as a JSON string"),
        args_schema=TestArgSchema,
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
        ),
    )
    expected = (
        '{"meta":null,"content":[{"type":"text","text":"Hello, world!","annotations":null,"meta":null}],'  # noqa: E501
        '"structuredContent":null,"isError":false}'
    )

    with patch(
        "portia.tool.get_mcp_session",
        new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
    ):
        tool_result = await tool.arun(get_test_tool_context(), a=1, b=2)
        assert tool_result == expected


@pytest.mark.asyncio
async def test_portia_mcp_tool_async_call_with_error() -> None:
    """Test invoking a tool via MCP asynchronously with error."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.call_tool.return_value = mcp.types.CallToolResult(
        content=[],
        isError=True,
    )

    class TestArgSchema(BaseModel):
        a: int
        b: int

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:test_tool",
        name="test_tool",
        description="I am a tool",
        output_schema=("str", "Tool output formatted as a JSON string"),
        args_schema=TestArgSchema,
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
        ),
    )

    with (
        patch(
            "portia.tool.get_mcp_session",
            new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
        ),
        pytest.raises(ToolHardError),
    ):
        await tool.arun(get_test_tool_context(), a=1, b=2)


@pytest.mark.asyncio
async def test_portia_mcp_tool_async_call_with_complex_response() -> None:
    """Test invoking a tool via MCP asynchronously with complex response."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.call_tool.return_value = mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(type="text", text="Result: 42"),
            mcp.types.TextContent(type="text", text="Additional info"),
        ],
        isError=False,
    )

    class TestArgSchema(BaseModel):
        query: str

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:complex_tool",
        name="complex_tool",
        description="A tool that returns complex content",
        output_schema=("str", "Tool output formatted as a JSON string"),
        args_schema=TestArgSchema,
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
        ),
    )

    with patch(
        "portia.tool.get_mcp_session",
        new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
    ):
        tool_result = await tool.arun(get_test_tool_context(), query="test query")
        # Verify the result contains both text content items
        assert "Result: 42" in tool_result
        assert "Additional info" in tool_result


@pytest.mark.asyncio
async def test_portia_mcp_tool_async_call_with_no_args() -> None:
    """Test invoking a tool via MCP asynchronously with no arguments."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.call_tool.return_value = mcp.types.CallToolResult(
        content=[mcp.types.TextContent(type="text", text="No args tool result")],
        isError=False,
    )

    class EmptyArgSchema(BaseModel):
        """Empty schema for tools with no arguments."""

    tool = PortiaMcpTool(
        id="mcp:mock_mcp:no_args_tool",
        name="no_args_tool",
        description="A tool that takes no arguments",
        output_schema=("str", "Tool output formatted as a JSON string"),
        args_schema=EmptyArgSchema,
        mcp_client_config=StdioMcpClientConfig(
            server_name="mock_mcp",
            command="test",
            args=["test"],
        ),
    )

    with patch(
        "portia.tool.get_mcp_session",
        new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
    ):
        tool_result = await tool.arun(get_test_tool_context())
        assert "No args tool result" in tool_result


def test_remote_tool_batch_ready_check(httpx_mock: HTTPXMock) -> None:
    """Test batch_ready_check classmethod."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/batch/ready/",
        json={"ready": True, "clarifications": []},
    )

    ctx = get_test_tool_context()
    config = get_test_config()

    # Configure mock for PortiaCloudClient to return our client
    mock_client = httpx.Client(base_url=endpoint)
    with patch("portia.cloud.PortiaCloudClient.new_client", return_value=mock_client):
        response = PortiaRemoteTool.batch_ready_check(
            config,
            {"tool1", "tool2"},
            ctx,
        )

    assert response.ready is True
    assert len(response.clarifications) == 0

    # Verify correct request was made
    request = httpx_mock.get_request(
        method="POST",
        url=f"{endpoint}/api/v0/tools/batch/ready/",
    )
    assert request is not None

    # Check request JSON
    json_data = request.read().decode()
    request_body = json.loads(json_data)
    assert request_body["tool_ids"] == ["tool1", "tool2"]
    assert request_body["execution_context"]["end_user_id"] == ctx.end_user.external_id
    assert request_body["execution_context"]["plan_run_id"] == str(ctx.plan_run.id)


def test_remote_tool_batch_ready_check_not_ready(httpx_mock: HTTPXMock) -> None:
    """Test batch_ready_check classmethod with tools not ready."""
    endpoint = "https://api.fake-portia.test"
    ctx = get_test_tool_context()

    # Create a clarification to include in the response
    clarification = ActionClarification(
        id=ClarificationUUID(),
        category=ClarificationCategory.ACTION,
        user_guidance="Please authenticate",
        action_url=HttpUrl("https://example.com"),
        plan_run_id=ctx.plan_run.id,
    )

    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/batch/ready/",
        json={"ready": False, "clarifications": [clarification.model_dump(mode="json")]},
    )

    config = get_test_config()
    # Configure mock for PortiaCloudClient to return our client
    mock_client = httpx.Client(base_url=endpoint)
    with patch("portia.cloud.PortiaCloudClient.new_client", return_value=mock_client):
        response = PortiaRemoteTool.batch_ready_check(
            config,
            {"tool1", "tool2"},
            ctx,
        )

    assert response.ready is False
    assert len(response.clarifications) == 1
    assert isinstance(response.clarifications[0], ActionClarification)
    assert response.clarifications[0] == clarification


def test_remote_tool_batch_ready_check_404_fallback(httpx_mock: HTTPXMock) -> None:
    """Test batch_ready_check classmethod with 404 fallback."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/batch/ready/",
        status_code=404,
        json={"error": "Resource not found", "status": 404},
    )

    ctx = get_test_tool_context()
    config = get_test_config()

    # Configure mock for PortiaCloudClient to return our client
    mock_client = httpx.Client(base_url=endpoint)
    with patch("portia.cloud.PortiaCloudClient.new_client", return_value=mock_client):
        response = PortiaRemoteTool.batch_ready_check(
            config,
            {"tool1", "tool2"},
            ctx,
        )

    assert response.ready is True
    assert len(response.clarifications) == 0


def test_structured_output_schema(add_tool: AdditionTool) -> None:
    """Test structured output schema."""
    assert add_tool.structured_output_schema is None

    class AdditionOutput(BaseModel):
        result: int

    class StructuredAdditionTool(AdditionTool):
        structured_output_schema: type[BaseModel] | None = AdditionOutput

        def run(self, _: ToolRunContext, a: int, b: int) -> int:
            return AdditionOutput(result=a + b)  # type: ignore[ReportReturnType]

    structured_add_tool = StructuredAdditionTool()
    assert structured_add_tool.structured_output_schema is AdditionOutput

    output = structured_add_tool._run(get_test_tool_context(), a=1, b=2)
    assert output is not None
    assert isinstance(output, LocalDataValue)
    assert output.value == AdditionOutput(result=3)


def test_structured_output_schema_coercion(add_tool: AdditionTool) -> None:
    """Test structured output schema."""
    assert add_tool.structured_output_schema is None

    class AdditionOutput(BaseModel):
        result: int

    class StructuredAdditionTool(AdditionTool):
        structured_output_schema: type[BaseModel] | None = AdditionOutput

        def run(self, _: ToolRunContext, a: int, b: int) -> int:
            return AdditionOutput(result=a + b)  # type: ignore[ReportReturnType]

    structured_add_tool = StructuredAdditionTool()
    assert structured_add_tool.structured_output_schema is AdditionOutput

    output = structured_add_tool._run(get_test_tool_context(), a=1, b=2)
    assert output is not None
    assert isinstance(output, LocalDataValue)
    assert output.value == AdditionOutput(result=3)


def test_structured_output_schema_coercion_error(add_tool: AdditionTool) -> None:
    """Test structured output schema."""
    assert add_tool.structured_output_schema is None

    class AdditionOutput(BaseModel):
        result: int

    class StructuredAdditionTool(AdditionTool):
        structured_output_schema: type[BaseModel] | None = AdditionOutput

        def run(self, _: ToolRunContext, a: int, b: int) -> int:  # noqa: ARG002
            return {"result": "not an int"}  # type: ignore[ReportReturnType]

    structured_add_tool = StructuredAdditionTool()
    assert structured_add_tool.structured_output_schema is AdditionOutput

    output = structured_add_tool._run(get_test_tool_context(), a=1, b=2)
    assert output is not None
    assert isinstance(output, LocalDataValue)
    assert output.value == {"result": "not an int"}
