"""Logging functions for managing and configuring loggers.

This module defines functions and classes to manage logging within the application. It provides a
`LoggerManager` class that manages the package-level logger and allows customization.
The `LoggerInterface` defines the general interface for loggers, and the default logger is provided
by `loguru`. The `logger` function returns the active logger, and the `LoggerManager` can be used
to configure logging behavior.

Classes in this file include:

- `LoggerInterface`: A protocol defining the common logging methods (`debug`, `info`, `warning`,
`error`, `critical`).
- `LoggerManager`: A class for managing the logger, allowing customization and configuration from
the application's settings.

This module ensures flexible and configurable logging, supporting both default and custom loggers.

"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol

from loguru import logger as default_logger

if TYPE_CHECKING:
    from portia.config import Config


class LoggerInterface(Protocol):
    """General Interface for loggers.

    This interface defines the common methods that any logger should implement. The methods are:

    - `debug`: For logging debug-level messages.
    - `info`: For logging informational messages.
    - `warning`: For logging warning messages.
    - `error`: For logging error messages.
    - `critical`: For logging critical error messages.

    These methods are used throughout the application for logging messages at various levels.

    """

    def debug(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def info(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def warning(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def error(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def critical(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102


DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | "
    "{extra}"
)


class LoggerManager:
    """Manages the package-level logger.

    The `LoggerManager` is responsible for initializing and managing the logger used throughout
    the application. It provides functionality to configure the logger, set a custom logger,
    and adjust logging settings based on the application's configuration.

    Args:
        custom_logger (LoggerInterface | None): A custom logger to be used. If not provided,
                                                 the default `loguru` logger will be used.

    Attributes:
        logger (LoggerInterface): The current active logger.
        custom_logger (bool): A flag indicating whether a custom logger is in use.

    Methods:
        logger: Returns the active logger.
        set_logger: Sets a custom logger.
        configure_from_config: Configures the logger based on the provided configuration.

    """

    def __init__(self, custom_logger: LoggerInterface | None = None) -> None:
        """Initialize the LoggerManager.

        Args:
            custom_logger (LoggerInterface | None): A custom logger to use. Defaults to None.

        """
        default_logger.remove()
        default_logger.add(
            sys.stdout,
            level="INFO",
            format=DEFAULT_LOG_FORMAT,
            serialize=False,
        )
        self._logger: LoggerInterface = custom_logger or default_logger  # type: ignore  # noqa: PGH003
        self.custom_logger = False

    @property
    def logger(self) -> LoggerInterface:
        """Get the current logger.

        Returns:
            LoggerInterface: The active logger being used.

        """
        return self._logger

    def set_logger(self, custom_logger: LoggerInterface) -> None:
        """Set a custom logger.

        Args:
            custom_logger (LoggerInterface): The custom logger to be used.

        """
        self._logger = custom_logger
        self.custom_logger = True

    def configure_from_config(self, config: Config) -> None:
        """Configure the global logger based on the library's configuration.

        This method configures the logger's log level and output sink based on the application's
        settings. If a custom logger is in use, it will skip the configuration and log a warning.

        Args:
            config (Config): The configuration object containing the logging settings.

        """
        if self.custom_logger:
            # Log a warning if a custom logger is being used
            self._logger.warning("Custom logger is in use; skipping log level configuration.")
        else:
            default_logger.remove()
            log_sink = config.default_log_sink
            match config.default_log_sink:
                case "sys.stdout":
                    log_sink = sys.stdout
                case "sys.stderr":
                    log_sink = sys.stderr

            default_logger.add(
                log_sink,
                level=config.default_log_level.value,
                format=DEFAULT_LOG_FORMAT,
                serialize=config.json_log_serialize,
            )


# Expose manager to allow updating logger
logger_manager = LoggerManager()


def logger() -> LoggerInterface:
    """Return the active logger.

    Returns:
        LoggerInterface: The current active logger being used.

    """
    return logger_manager.logger
