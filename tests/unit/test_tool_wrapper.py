"""Tests for the ToolCallWrapper class."""

import pytest

from portia.clarification import Clarification
from portia.end_user import EndUser
from portia.errors import ToolHardError
from portia.execution_agents.output import LocalOutput
from portia.storage import AdditionalStorage, ToolCallRecord, ToolCallStatus
from portia.tool import Tool
from portia.tool_wrapper import ToolCallWrapper
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    ErrorTool,
    NoneTool,
    get_test_plan_run,
    get_test_tool_context,
)


class MockStorage(AdditionalStorage):
    """Mock implementation of AdditionalStorage for testing."""

    def __init__(self) -> None:
        """Save records in array."""
        self.records = []
        self.end_users = {}

    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Save records in array."""
        self.records.append(tool_call)

    def save_end_user(self, end_user: EndUser) -> EndUser:
        """Add end_user to dict.

        Args:
            end_user (EndUser): The EndUser object to save.

        """
        self.end_users[end_user.external_id] = end_user
        return end_user

    def get_end_user(self, end_user_id: str) -> EndUser:
        """Get end_user from dict or init a new one.

        Args:
            end_user_id (str): The id of the end user object to get.

        """
        if end_user_id in self.end_users:
            return self.end_users[end_user_id]
        end_user = EndUser(external_id=end_user_id)
        return self.save_end_user(end_user)


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
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(child_tool=mock_tool, storage=mock_storage, plan_run=plan_run)
    assert wrapper.name == mock_tool.name
    assert wrapper.description == mock_tool.description


def test_tool_call_wrapper_run_success(mock_tool: Tool, mock_storage: MockStorage) -> None:
    """Test successful run of the ToolCallWrapper."""
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(mock_tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    result = wrapper.run(ctx, 1, 2)
    assert result == 3
    assert mock_storage.records[-1].status == ToolCallStatus.SUCCESS


def test_tool_call_wrapper_run_with_exception(
    mock_storage: MockStorage,
) -> None:
    """Test run of the ToolCallWrapper when the child tool raises an exception."""
    tool = ErrorTool()
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError, match="Test error"):
        wrapper.run(ctx, "Test error", False, False)  # noqa: FBT003
    assert mock_storage.records[-1].status == ToolCallStatus.FAILED


def test_tool_call_wrapper_run_with_clarification(
    mock_storage: MockStorage,
) -> None:
    """Test run of the ToolCallWrapper when the child tool returns a Clarification."""
    (_, plan_run) = get_test_plan_run()
    tool = ClarificationTool()
    wrapper = ToolCallWrapper(tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    result = wrapper.run(ctx, "new clarification")
    assert isinstance(result, Clarification)
    assert mock_storage.records[-1].status == ToolCallStatus.NEED_CLARIFICATION


def test_tool_call_wrapper_run_records_latency(mock_tool: Tool, mock_storage: MockStorage) -> None:
    """Test that the ToolCallWrapper records latency correctly."""
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(mock_tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    wrapper.run(ctx, 1, 2)
    assert mock_storage.records[-1].latency_seconds > 0


def test_tool_call_wrapper_run_returns_none(mock_storage: MockStorage) -> None:
    """Test that the ToolCallWrapper records latency correctly."""
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(NoneTool(), mock_storage, plan_run)
    ctx = get_test_tool_context()
    wrapper.run(ctx)
    assert mock_storage.records[-1].output
    assert mock_storage.records[-1].output == LocalOutput(value=None).model_dump(mode="json")
