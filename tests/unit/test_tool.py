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
from portia.mcp_session import StdioMcpClientConfig
from portia.tool import PortiaMcpTool, PortiaRemoteTool
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
        description: str = "this is a description" * 250

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
        result = tool._run_async(  # noqa: F841, SLF001
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


def test_remote_tool_soft_error(httpx_mock: HTTPXMock) -> None:
    """Test remote soft errors come back to soft errors."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={"output": {"value": "ToolSoftError: An error occurred."}},
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


def test_remote_tool_hard_error(httpx_mock: HTTPXMock) -> None:
    """Test remote hard errors come back to hard errors."""
    endpoint = "https://api.fake-portia.test"
    httpx_mock.add_response(
        url=f"{endpoint}/api/v0/tools/test/run/",
        json={"output": {"value": "ToolHardError: An error occurred."}},
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
def test_remote_tool_ready(httpx_mock: HTTPXMock, response_json: dict, is_ready: bool) -> None:  # noqa: FBT001
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
def test_remote_tool_ready_error(httpx_mock: HTTPXMock, status_code: int, is_ready: bool) -> None:  # noqa: FBT001
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
    with patch("portia.cloud.PortiaCloudClient.get_client", return_value=mock_client):
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
    with patch("portia.cloud.PortiaCloudClient.get_client", return_value=mock_client):
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
    with patch("portia.cloud.PortiaCloudClient.get_client", return_value=mock_client):
        response = PortiaRemoteTool.batch_ready_check(
            config,
            {"tool1", "tool2"},
            ctx,
        )

    assert response.ready is True
    assert len(response.clarifications) == 0
