"""Tests for execution hooks."""

from unittest.mock import MagicMock

from portia.portia import ExecutionHooks


def test_execution_hooks_initialization() -> None:
    """Test that the ExecutionHooks class can be initialized with all hooks."""
    clarification_handler_mock = MagicMock()
    before_first_step_mock = MagicMock()
    before_step_mock = MagicMock()
    after_step_mock = MagicMock()
    after_last_step_mock = MagicMock()

    execution_hooks = ExecutionHooks(
        clarification_handler=clarification_handler_mock,
        before_first_step=before_first_step_mock,
        before_step=before_step_mock,
        after_step=after_step_mock,
        after_last_step=after_last_step_mock,
    )

    assert execution_hooks.clarification_handler == clarification_handler_mock
    assert execution_hooks.before_first_step == before_first_step_mock
    assert execution_hooks.before_step == before_step_mock
    assert execution_hooks.after_step == after_step_mock
    assert execution_hooks.after_last_step == after_last_step_mock


def test_execution_hooks_default_values() -> None:
    """Test that the ExecutionHooks class initializes with default values."""
    execution_hooks = ExecutionHooks()

    assert execution_hooks.clarification_handler is None
    assert execution_hooks.before_first_step is None
    assert execution_hooks.before_step is None
    assert execution_hooks.after_step is None
    assert execution_hooks.after_last_step is None


def test_execution_hooks_partial_initialization() -> None:
    """Test that the ExecutionHooks class can be initialized with some hooks."""
    before_first_step_mock = MagicMock()
    after_last_step_mock = MagicMock()

    execution_hooks = ExecutionHooks(
        before_first_step=before_first_step_mock,
        after_last_step=after_last_step_mock,
    )

    assert execution_hooks.before_first_step == before_first_step_mock
    assert execution_hooks.after_last_step == after_last_step_mock

    assert execution_hooks.clarification_handler is None
    assert execution_hooks.before_step is None
    assert execution_hooks.after_step is None
