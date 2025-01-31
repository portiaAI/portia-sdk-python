"""Tests for the ToolCallWrapper class."""

import pytest

from portia.clarification import Clarification
from portia.errors import ToolHardError
from portia.storage import ToolCallRecord, ToolCallStatus, ToolCallStorage
from portia.tool import Tool
from portia.tool_wrapper import ToolCallWrapper
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    ErrorTool,
    get_test_tool_context,
    get_test_workflow,
)


class MockStorage(ToolCallStorage):
    """Mock implementation of ToolCallStorage for testing."""

    def __init__(self) -> None:
        """Save records in array."""
        self.records = []

    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Save records in array."""
        self.records.append(tool_call)


@pytest.fixture
def mock_tool() -> Tool:
    """Fixture to create a mock tool instance."""
    return AdditionTool()


@pytest.fixture
def mock_storage() -> MockStorage:
    """Fixture to create a mock storage instance."""
    return MockStorage()


def test_tool_call_wrapper_initialization(mock_tool: Tool, mock_storage: MockStorage) -> None:
    """Test initialization of the ToolCallWrapper."""
    (_, workflow) = get_test_workflow()
    wrapper = ToolCallWrapper(child_tool=mock_tool, storage=mock_storage, workflow=workflow)
    assert wrapper.name == mock_tool.name
    assert wrapper.description == mock_tool.description


def test_tool_call_wrapper_run_success(mock_tool: Tool, mock_storage: MockStorage) -> None:
    """Test successful run of the ToolCallWrapper."""
    (_, workflow) = get_test_workflow()
    wrapper = ToolCallWrapper(mock_tool, mock_storage, workflow)
    ctx = get_test_tool_context()
    result = wrapper.run(ctx, 1, 2)
    assert result == 3
    assert mock_storage.records[-1].status == ToolCallStatus.SUCCESS


def test_tool_call_wrapper_run_with_exception(
    mock_storage: MockStorage,
) -> None:
    """Test run of the ToolCallWrapper when the child tool raises an exception."""
    tool = ErrorTool()
    (_, workflow) = get_test_workflow()
    wrapper = ToolCallWrapper(tool, mock_storage, workflow)
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError, match="Test error"):
        wrapper.run(ctx, "Test error", False, False)  # noqa: FBT003
    assert mock_storage.records[-1].status == ToolCallStatus.FAILED


def test_tool_call_wrapper_run_with_clarification(
    mock_storage: MockStorage,
) -> None:
    """Test run of the ToolCallWrapper when the child tool returns a Clarification."""
    (_, workflow) = get_test_workflow()
    tool = ClarificationTool()
    wrapper = ToolCallWrapper(tool, mock_storage, workflow)
    ctx = get_test_tool_context()
    ctx.execution_context.additional_data["raise_clarification"] = "True"
    result = wrapper.run(ctx, "new clarification")
    assert isinstance(result, Clarification)
    assert mock_storage.records[-1].status == ToolCallStatus.NEED_CLARIFICATION


def test_tool_call_wrapper_run_records_latency(mock_tool: Tool, mock_storage: MockStorage) -> None:
    """Test that the ToolCallWrapper records latency correctly."""
    (_, workflow) = get_test_workflow()
    wrapper = ToolCallWrapper(mock_tool, mock_storage, workflow)
    ctx = get_test_tool_context()
    wrapper.run(ctx, 1, 2)
    assert mock_storage.records[-1].latency_seconds > 0
