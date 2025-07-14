"""Tests for logging functions."""

from unittest.mock import Mock

import pytest

from portia.config import LogLevel
from portia.logger import (
    FUNCTION_COLOR_MAP,
    Formatter,
    LoggerInterface,
    LoggerManager,
    SafeLogger,
    logger,
    logger_manager,
)


@pytest.mark.parametrize(
    ("record", "expected_color"),
    [
        (
            {"name": "portia.portia", "function": "_execute_plan_run"},
            FUNCTION_COLOR_MAP["run"],
        ),
        (
            {"name": "portia.portia", "function": "_handle_introspection_outcome"},
            FUNCTION_COLOR_MAP["introspection"],
        ),
        (
            {"name": "portia.storage", "function": "save_tool_call"},
            FUNCTION_COLOR_MAP["tool"],
        ),
        (
            {"name": "portia.portia", "function": "plan"},
            FUNCTION_COLOR_MAP["plan"],
        ),
        (
            {"name": "portia.portia", "function": "_raise_clarifications"},
            FUNCTION_COLOR_MAP["clarification"],
        ),
        (
            {"name": "portia.tool_wrapper", "function": "run"},
            FUNCTION_COLOR_MAP["tool"],
        ),
    ],
)
def test_logger_formatter_get_function_color(record: dict, expected_color: str) -> None:
    """Test the logger formatter get_function_color method."""
    logger_formatter = Formatter()
    assert logger_formatter._get_function_color_(record) == expected_color


def test_logger_sanitize_message() -> None:
    """Test the logger sanitize_message method."""
    logger_formatter = Formatter()
    assert logger_formatter._sanitize_message_("<test>") == r"\<test\>"
    assert logger_formatter._sanitize_message_("{test} {{test}}") == "{{test}} {{test}}"
    assert logger_formatter._sanitize_message_('{"test": "<test>"}') == '{{"test": "\\<test\\>"}}'

    # a long message gets truncated correctly
    long_message = "test\n" * 100
    truncated_message = logger_formatter._sanitize_message_(long_message)
    assert len(truncated_message.split("\n")) == logger_formatter.max_lines
    assert truncated_message.endswith("test\n")
    assert truncated_message.startswith("test\n")


def test_logger_manager_initialization() -> None:
    """Test initialization of LoggerManager with default logger."""
    logger_manager = LoggerManager()
    assert logger_manager.custom_logger is False


def test_logger_manager_with_custom_logger() -> None:
    """Test initialization of LoggerManager with a custom logger."""
    mock_logger = Mock(spec=LoggerInterface)
    logger_manager = LoggerManager(custom_logger=mock_logger)

    assert logger_manager.logger == mock_logger
    assert logger_manager.custom_logger is False


def test_set_logger() -> None:
    """Test setting a custom logger."""
    logger_manager = LoggerManager()
    mock_logger = Mock(spec=LoggerInterface)

    logger_manager.set_logger(mock_logger)
    assert logger_manager.logger == mock_logger
    assert logger_manager.custom_logger is True


def test_configure_from_config() -> None:
    """Test configuring the logger from a Config instance."""
    logger_manager = LoggerManager()
    mock_config = Mock(
        default_log_sink="sys.stdout",
        default_log_level=LogLevel.DEBUG,
        json_log_serialize=False,
    )

    logger_manager.configure_from_config(mock_config)

    # Verify log level and sink configuration
    assert mock_config.default_log_level == LogLevel.DEBUG
    assert mock_config.default_log_sink == "sys.stdout"


def test_configure_from_config_stderr() -> None:
    """Test configuring the logger from a Config instance."""
    logger_manager = LoggerManager()
    mock_config = Mock(
        default_log_sink="sys.stderr",
        default_log_level=LogLevel.INFO,
        json_log_serialize=False,
    )

    logger_manager.configure_from_config(mock_config)

    # Verify log level and sink configuration
    assert mock_config.default_log_level == LogLevel.INFO
    assert mock_config.default_log_sink == "sys.stderr"


def test_configure_from_config_custom_logger() -> None:
    """Test warning when configuring logger with a custom logger set."""
    mock_logger = Mock(spec=LoggerInterface)
    logger_manager = LoggerManager(custom_logger=mock_logger)
    logger_manager.set_logger(mock_logger)

    mock_config = Mock(
        default_log_sink="sys.stderr",
        default_log_level="INFO",
        json_log_serialize=True,
    )

    logger_manager.configure_from_config(mock_config)
    mock_logger.warning.assert_called_once_with(
        "Custom logger is in use; skipping log level configuration.",
    )


def test_logger() -> None:
    """Test the LoggerProxy provides access to the current logger."""
    mock_logger = Mock(spec=LoggerInterface)
    logger_manager.set_logger(mock_logger)

    assert logger() == mock_logger


def test_safe_logger_successful_logs() -> None:
    """Test SafeLogger successfully passes through logs to child logger."""
    mock_logger = Mock(spec=LoggerInterface)
    safe_logger = SafeLogger(mock_logger)

    # Test each log level
    safe_logger.debug("debug message", "arg1", kwarg1="value1")  # noqa: PLE1205
    safe_logger.info("info message", "arg1", kwarg1="value1")  # noqa: PLE1205
    safe_logger.warning("warning message", "arg1", kwarg1="value1")  # noqa: PLE1205
    safe_logger.error("error message", "arg1", kwarg1="value1")  # noqa: PLE1205
    safe_logger.critical("critical message", "arg1", kwarg1="value1")  # noqa: PLE1205
    safe_logger.exception("exception message", "arg1", kwarg1="value1")  # noqa: PLE1205

    # Verify each method was called with correct arguments
    mock_logger.debug.assert_called_once_with("debug message", "arg1", kwarg1="value1")
    mock_logger.info.assert_called_once_with("info message", "arg1", kwarg1="value1")
    mock_logger.warning.assert_called_once_with("warning message", "arg1", kwarg1="value1")
    mock_logger.error.assert_called_once_with("error message", "arg1", kwarg1="value1")
    mock_logger.critical.assert_called_once_with("critical message", "arg1", kwarg1="value1")
    mock_logger.exception.assert_called_once_with("exception message", "arg1", kwarg1="value1")


def test_safe_logger_error_handling() -> None:
    """Test SafeLogger catches and logs exceptions from child logger."""
    mock_logger = Mock(spec=LoggerInterface)
    safe_logger = SafeLogger(mock_logger)

    # Make each method raise an exception
    mock_logger.debug.side_effect = Exception("debug error")
    mock_logger.info.side_effect = Exception("info error")
    mock_logger.warning.side_effect = Exception("warning error")
    mock_logger.critical.side_effect = Exception("critical error")
    mock_logger.exception.side_effect = Exception("exception error")
    # Test error log separately as all the other methods call the error log on exception
    mock_logger.error.side_effect = None

    safe_logger.debug("debug message")
    safe_logger.info("info message")
    safe_logger.warning("warning message")
    safe_logger.critical("critical message")
    safe_logger.exception("exception message")

    mock_logger.error.side_effect = [Exception("error error"), None]
    safe_logger.error("error message")

    assert mock_logger.debug.call_count == 1  # Original call + error log
    assert mock_logger.info.call_count == 1
    assert mock_logger.warning.call_count == 1
    assert mock_logger.critical.call_count == 1
    assert mock_logger.exception.call_count == 1
    assert mock_logger.error.call_count == 7

    mock_logger.error.assert_any_call("Failed to log: debug error")
    mock_logger.error.assert_any_call("Failed to log: info error")
    mock_logger.error.assert_any_call("Failed to log: warning error")
    mock_logger.error.assert_any_call("Failed to log: error error")
    mock_logger.error.assert_any_call("Failed to log: critical error")
    mock_logger.error.assert_any_call("Failed to log: exception error")
