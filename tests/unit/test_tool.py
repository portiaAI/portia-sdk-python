"""Tests for the Tool class."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import Response
from pydantic import SecretStr

from portia.context import empty_context, get_execution_context
from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.tool import PortiaRemoteTool
from tests.utils import AdditionTool, ClarificationTool, ErrorTool


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
    assert add_tool.description == "Takes two numbers and adds them together"


def test_tool_initialization_long_description() -> None:
    """Test initialization of a Tool."""

    class FakeAdditionTool(AdditionTool):
        description: str = "this is a description" * 100

    with pytest.raises(InvalidToolDescriptionError):
        FakeAdditionTool()


def test_tool_to_langchain() -> None:
    """Test langchain rep of a Tool."""
    tool = AdditionTool()
    tool.to_langchain(ctx=empty_context(), return_artifact=False)


def test_run_method(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    ctx = get_execution_context()
    result = add_tool.run(ctx, a, b)
    assert result == a + b


def test_handle(add_tool: AdditionTool) -> None:
    """Test the run method of the AddTool."""
    a, b = 1, 2
    ctx = get_execution_context()
    result = add_tool.run(ctx, a, b)
    assert result == a + b


def test_run_method_with_uncaught_error() -> None:
    """Test the _run method wraps errors."""
    tool = ErrorTool()
    with pytest.raises(ToolSoftError):
        tool._run(  # noqa: SLF001
            ctx=empty_context(),
            error_str="this is an error",
            return_uncaught_error=True,
            return_soft_error=False,
        )


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
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=Exception(),
    )
    mock_response.json = MagicMock(
        return_value={"output": {"value": "An error occurred."}},
    )
    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
    ):
        tool = PortiaRemoteTool(
            id="test",
            name="test",
            description="",
            output_schema=("", ""),
            api_key=SecretStr(""),
            api_endpoint="https://example.com",
        )
        with pytest.raises(ToolHardError):
            tool.run(empty_context())

        mock_post.assert_called_once_with(
            url="https://example.com/api/v0/tools/test/run/",
            content='{"arguments": {}, "execution_context": {"end_user_id": "", "additional_data": {}}}',  # noqa: E501
            headers={
                "Authorization": "Api-Key ",
                "Content-Type": "application/json",
            },
            timeout=60,
        )


def test_remote_tool_soft_error() -> None:
    """Test remote soft errors come back to soft errors."""
    mock_response = MagicMock(spec=Response)
    mock_response.is_success = True
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"output": {"value": "ToolSoftError: An error occurred."}},
    )
    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
    ):
        tool = PortiaRemoteTool(
            id="test",
            name="test",
            description="",
            output_schema=("", ""),
            api_key=SecretStr(""),
            api_endpoint="https://example.com",
        )
        with pytest.raises(ToolSoftError):
            tool.run(empty_context())

        mock_post.assert_called_once_with(
            url="https://example.com/api/v0/tools/test/run/",
            content='{"arguments": {}, "execution_context": {"end_user_id": "", "additional_data": {}}}',  # noqa: E501
            headers={
                "Authorization": "Api-Key ",
                "Content-Type": "application/json",
            },
            timeout=60,
        )


def test_remote_tool_hard_error() -> None:
    """Test remote hard errors come back to hard errors."""
    mock_response = MagicMock(spec=Response)
    mock_response.is_success = True
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"output": {"value": "ToolHardError: An error occurred."}},
    )
    with (
        patch("httpx.post", return_value=mock_response) as mock_post,
    ):
        tool = PortiaRemoteTool(
            id="test",
            name="test",
            description="",
            output_schema=("", ""),
            api_key=SecretStr(""),
            api_endpoint="https://example.com",
        )
        with pytest.raises(ToolHardError):
            tool.run(empty_context())

        mock_post.assert_called_once_with(
            url="https://example.com/api/v0/tools/test/run/",
            content='{"arguments": {}, "execution_context": {"end_user_id": "", "additional_data": {}}}',  # noqa: E501
            headers={
                "Authorization": "Api-Key ",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
