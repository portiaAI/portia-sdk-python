"""Tests for logging functions."""

from unittest.mock import Mock

import pytest

from portia.errors import InvalidLoggerError
from portia.logging import logger_manager, LoggerInterface, LoggerManager, LoggerProxy


def test_logger_manager_initialization() -> None:
    """Test initialization of LoggerManager with default logger."""
    logger_manager = LoggerManager()
    assert isinstance(logger_manager.logger, LoggerInterface)
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


def test_set_logger_invalid_logger() -> None:
    """Test setting an invalid logger raises an error."""
    logger_manager = LoggerManager()

    with pytest.raises(InvalidLoggerError):
        logger_manager.set_logger(Mock())


def test_configure_from_config() -> None:
    """Test configuring the logger from a Config instance."""
    logger_manager = LoggerManager()
    mock_config = Mock(
        default_log_sink="sys.stdout",
        default_log_level="DEBUG",
        json_log_serialize=False,
    )

    logger_manager.configure_from_config(mock_config)

    # Verify log level and sink configuration
    assert mock_config.default_log_level.upper() == "DEBUG"
    assert mock_config.default_log_sink == "sys.stdout"


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


def test_logger_proxy() -> None:
    """Test the LoggerProxy provides access to the current logger."""
    mock_logger = Mock(spec=LoggerInterface)
    logger_manager.set_logger(mock_logger)
    logger_proxy = LoggerProxy()

    assert logger_proxy.logger == mock_logger
