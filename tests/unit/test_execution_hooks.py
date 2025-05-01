"""Tests for execution hooks."""

from unittest.mock import MagicMock

from portia.portia import ExecutionHooks


def test_execution_hooks_initialization() -> None:
    """Test that the ExecutionHooks class can be initialized with all hooks."""
    # Create mock hooks
    clarification_handler_mock = MagicMock()
    before_first_tool_call_mock = MagicMock()
    before_tool_call_mock = MagicMock()
    after_tool_call_mock = MagicMock()
    after_last_tool_call_mock = MagicMock()

    # Create execution hooks with all mocks
    execution_hooks = ExecutionHooks(
        clarification_handler=clarification_handler_mock,
        before_first_tool_call=before_first_tool_call_mock,
        before_tool_call=before_tool_call_mock,
        after_tool_call=after_tool_call_mock,
        after_last_tool_call=after_last_tool_call_mock,
    )

    # Assert that all hooks are set correctly
    assert execution_hooks.clarification_handler == clarification_handler_mock
    assert execution_hooks.before_first_tool_call == before_first_tool_call_mock
    assert execution_hooks.before_tool_call == before_tool_call_mock
    assert execution_hooks.after_tool_call == after_tool_call_mock
    assert execution_hooks.after_last_tool_call == after_last_tool_call_mock


def test_execution_hooks_default_values() -> None:
    """Test that the ExecutionHooks class initializes with default values."""
    # Create execution hooks with no arguments
    execution_hooks = ExecutionHooks()

    # Assert that all hooks are None by default
    assert execution_hooks.clarification_handler is None
    assert execution_hooks.before_first_tool_call is None
    assert execution_hooks.before_tool_call is None
    assert execution_hooks.after_tool_call is None
    assert execution_hooks.after_last_tool_call is None


def test_execution_hooks_partial_initialization() -> None:
    """Test that the ExecutionHooks class can be initialized with some hooks."""
    # Create mock hooks
    before_first_tool_call_mock = MagicMock()
    after_last_tool_call_mock = MagicMock()

    # Create execution hooks with some mocks
    execution_hooks = ExecutionHooks(
        before_first_tool_call=before_first_tool_call_mock,
        after_last_tool_call=after_last_tool_call_mock,
    )

    # Assert that specified hooks are set correctly
    assert execution_hooks.before_first_tool_call == before_first_tool_call_mock
    assert execution_hooks.after_last_tool_call == after_last_tool_call_mock

    # Assert that unspecified hooks are None
    assert execution_hooks.clarification_handler is None
    assert execution_hooks.before_tool_call is None
    assert execution_hooks.after_tool_call is None
