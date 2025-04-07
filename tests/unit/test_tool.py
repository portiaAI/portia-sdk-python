"""Tests for the Tool class."""

import json
from enum import Enum
from unittest.mock import MagicMock, patch

import httpx
import mcp
import pytest
from mcp import ClientSession
from pydantic import BaseModel, HttpUrl

from portia.clarification import (
    ActionClarification,
    ClarificationUUID,
    InputClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
)
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.mcp_session import StdioMcpClientConfig
from portia.tool import PortiaMcpTool, PortiaRemoteTool
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    ErrorTool,
    MockMcpSessionWrapper,
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
        description: str = "this is a description" * 100

    with pytest.raises(InvalidToolDescriptionError):
        FakeAdditionTool()


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
        tool._run(  # noqa: SLF001
            ctx=get_test_tool_context(),
            error_str="this is an error",
            return_uncaught_error=True,
            return_soft_error=False,
        )


def test_ready() -> None:
    """Test the ready method."""
    tool = ErrorTool()
    assert tool.ready(get_test_tool_context())


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


def test_remote_tool_hard_error_from_server() -> None:
    """Test http errors come back to hard errors."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception()
    mock_response.json.return_value = {"output": {"value": "An error occurred."}}

    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_soft_error() -> None:
    """Test remote soft errors come back to soft errors."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json = MagicMock(
        return_value={"output": {"value": "ToolSoftError: An error occurred."}},
    )
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    with pytest.raises(ToolSoftError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }
    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_bad_response() -> None:
    """Test remote soft errors come back to soft errors."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {"ot": {"value": "An error occurred."}}
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_hard_error() -> None:
    """Test remote hard errors come back to hard errors."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {"output": {"value": "ToolHardError: An error occurred."}}
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError):
        tool.run(ctx)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }
    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_ready() -> None:
    """Test remote tool ready."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json = MagicMock(
        return_value={"success": "true"},
    )
    mock_client.post.return_value = mock_response
    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )
    ctx = get_test_tool_context()
    assert tool.ready(ctx)

    content = {
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/ready/",
        content=json.dumps(content),
    )


def test_remote_tool_ready_error() -> None:
    """Test remote tool ready."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception()
    mock_response.json = MagicMock(
        return_value={"success": "true"},
    )
    mock_client.post.return_value = mock_response
    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    assert not tool.ready(ctx)

    content = {
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/ready/",
        content=json.dumps(content),
    )


def test_remote_tool_action_clarifications() -> None:
    """Test action clarifications."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json = MagicMock(
        return_value={
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
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )
    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, ActionClarification)
    assert output.action_url == HttpUrl("https://example.com")

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_input_clarifications() -> None:
    """Test Input clarifications."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json = MagicMock(
        return_value={
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
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, InputClarification)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_mc_clarifications() -> None:
    """Test Multi Choice clarifications."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json = MagicMock(
        return_value={
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
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, MultipleChoiceClarification)
    assert output.options == [1]

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
    )


def test_remote_tool_value_confirm_clarifications() -> None:
    """Test value confirm clarifications."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json = MagicMock(
        return_value={
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
    mock_client.post.return_value = mock_response

    tool = PortiaRemoteTool(
        id="test",
        name="test",
        description="",
        output_schema=("", ""),
        client=mock_client,
    )

    ctx = get_test_tool_context()
    output = tool.run(ctx)
    assert output is not None
    assert isinstance(output, ValueConfirmationClarification)

    content = {
        "arguments": {},
        "execution_context": {
            "end_user_id": ctx.execution_context.end_user_id or "",
            "plan_run_id": str(ctx.plan_run_id),
            "additional_data": ctx.execution_context.additional_data or {},
        },
    }

    mock_client.post.assert_called_once_with(
        url="/api/v0/tools/test/run/",
        content=json.dumps(content),
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
        '{"meta":null,"content":[{"type":"text","text":"Hello, world!","annotations":null}],'
        '"isError":false}'
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
