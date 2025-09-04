"""Tests for the ToolCallWrapper class."""

import pytest

from portia.clarification import Clarification
from portia.end_user import EndUser
from portia.errors import ToolHardError, ToolNotFoundError
from portia.execution_agents.output import LocalDataValue
from portia.open_source_tools.calculator_tool import CalculatorTool
from portia.open_source_tools.llm_tool import LLMTool
from portia.storage import AdditionalStorage, InMemoryStorage, ToolCallRecord, ToolCallStatus
from portia.tool import Tool
from portia.tool_registry import DefaultToolRegistry, InMemoryToolRegistry
from portia.tool_wrapper import ToolCallWrapper
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    ErrorTool,
    NoneTool,
    get_test_config,
    get_test_plan_run,
    get_test_tool_context,
)


class MockStorage(AdditionalStorage):
    """Mock implementation of AdditionalStorage for testing."""

    def __init__(self) -> None:
        """Save records in array."""
        self.records = []
        self.end_users = {}
        self.async_save_calls = []
        self.async_end_user_calls = []

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

    def get_end_user(self, external_id: str) -> EndUser:
        """Get end_user from dict or init a new one.

        Args:
            external_id (str): The id of the end user object to get.

        """
        if external_id in self.end_users:
            return self.end_users[external_id]
        end_user = EndUser(external_id=external_id)
        return self.save_end_user(end_user)

    async def asave_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Async save records in array."""
        self.async_save_calls.append(tool_call)
        self.records.append(tool_call)

    async def asave_end_user(self, end_user: EndUser) -> EndUser:
        """Async add end_user to dict."""
        self.async_end_user_calls.append(end_user)
        self.end_users[end_user.external_id] = end_user
        return end_user


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
    assert isinstance(result, list)
    assert isinstance(result[0], Clarification)
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
    assert mock_storage.records[-1].output == LocalDataValue(value=None).model_dump(mode="json")


# Async tests for ToolCallWrapper.arun method
@pytest.mark.asyncio
async def test_tool_call_wrapper_arun_success(mock_tool: Tool, mock_storage: MockStorage) -> None:
    """Test successful async run of the ToolCallWrapper."""
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(mock_tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    result = await wrapper.arun(ctx, 1, 2)
    assert result == 3

    # Wait for background tasks to complete
    import asyncio

    await asyncio.sleep(0.01)

    # Check that async storage methods were called
    assert len(mock_storage.async_save_calls) == 1
    assert mock_storage.async_save_calls[0].status == ToolCallStatus.SUCCESS
    assert len(mock_storage.async_end_user_calls) == 1


@pytest.mark.asyncio
async def test_tool_call_wrapper_arun_with_exception(mock_storage: MockStorage) -> None:
    """Test async run of the ToolCallWrapper when the child tool raises an exception."""
    tool = ErrorTool()
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError, match="Test error"):
        await wrapper.arun(ctx, "Test error", False, False)  # noqa: FBT003

    # Wait for background tasks to complete
    import asyncio

    await asyncio.sleep(0.01)

    # Check that async storage methods were called
    assert len(mock_storage.async_save_calls) == 1
    assert mock_storage.async_save_calls[0].status == ToolCallStatus.FAILED
    assert len(mock_storage.async_end_user_calls) == 1


@pytest.mark.asyncio
async def test_tool_call_wrapper_arun_with_clarification(mock_storage: MockStorage) -> None:
    """Test async run of the ToolCallWrapper when the child tool returns a Clarification."""
    (_, plan_run) = get_test_plan_run()
    tool = ClarificationTool()
    wrapper = ToolCallWrapper(tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    result = await wrapper.arun(ctx, "new clarification")
    assert isinstance(result, list)
    assert isinstance(result[0], Clarification)

    # Wait for background tasks to complete
    import asyncio

    await asyncio.sleep(0.01)

    # Check that async storage methods were called
    assert len(mock_storage.async_save_calls) == 1
    assert mock_storage.async_save_calls[0].status == ToolCallStatus.NEED_CLARIFICATION
    assert len(mock_storage.async_end_user_calls) == 1


@pytest.mark.asyncio
async def test_tool_call_wrapper_arun_records_latency(
    mock_tool: Tool, mock_storage: MockStorage
) -> None:
    """Test that the ToolCallWrapper records latency correctly in async mode."""
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(mock_tool, mock_storage, plan_run)
    ctx = get_test_tool_context()
    await wrapper.arun(ctx, 1, 2)

    # Wait for background tasks to complete
    import asyncio

    await asyncio.sleep(0.01)

    # Check that async storage methods were called
    assert len(mock_storage.async_save_calls) == 1
    assert mock_storage.async_save_calls[0].latency_seconds > 0
    assert len(mock_storage.async_end_user_calls) == 1


@pytest.mark.asyncio
async def test_tool_call_wrapper_arun_returns_none(mock_storage: MockStorage) -> None:
    """Test that the ToolCallWrapper handles None returns correctly in async mode."""
    (_, plan_run) = get_test_plan_run()
    wrapper = ToolCallWrapper(NoneTool(), mock_storage, plan_run)
    ctx = get_test_tool_context()
    await wrapper.arun(ctx)

    # Wait for background tasks to complete
    import asyncio

    await asyncio.sleep(0.01)

    # Check that async storage methods were called
    assert len(mock_storage.async_save_calls) == 1
    assert mock_storage.async_save_calls[0].output
    expected_output = LocalDataValue(value=None).model_dump(mode="json")
    assert mock_storage.async_save_calls[0].output == expected_output
    assert len(mock_storage.async_end_user_calls) == 1


def test_get_tool_in_registry() -> None:
    """Test retrieval of a tool in a registry."""
    _, plan_run = get_test_plan_run()
    tool = ToolCallWrapper.from_tool_id(
        CalculatorTool().id,
        DefaultToolRegistry(get_test_config()),
        InMemoryStorage(),
        plan_run,
    )
    assert tool is not None
    assert isinstance(tool._child_tool, CalculatorTool)


def test_portia_get_tool_for_step_none_tool_id() -> None:
    """Test that when step.tool_id is None, LLMTool is used as fallback."""
    _, plan_run = get_test_plan_run()
    tool = ToolCallWrapper.from_tool_id(
        None, DefaultToolRegistry(get_test_config()), InMemoryStorage(), plan_run
    )
    assert tool is None


def test_get_llm_tool_not_in_registry() -> None:
    """Test special case retrieval of LLMTool as it isn't explicitly in most tool registries."""
    _, plan_run = get_test_plan_run()
    tool = ToolCallWrapper.from_tool_id(
        LLMTool.LLM_TOOL_ID, InMemoryToolRegistry.from_local_tools([]), InMemoryStorage(), plan_run
    )
    assert tool is not None
    assert isinstance(tool._child_tool, LLMTool)


def test_get_tool_not_in_registry() -> None:
    """Test special case retrieval of LLMTool as it isn't explicitly in most tool registries."""
    _, plan_run = get_test_plan_run()
    with pytest.raises(ToolNotFoundError):
        ToolCallWrapper.from_tool_id(
            "not_in_registry",
            InMemoryToolRegistry.from_local_tools([]),
            InMemoryStorage(),
            plan_run,
        )
