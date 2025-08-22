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

import re
import sys
import traceback
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger as default_logger

if TYPE_CHECKING:
    from portia.config import Config

FUNCTION_COLOR_MAP = {
    "tool": "fg 87",
    "clarification": "fg 87",
    "introspection": "fg 87",
    "run": "fg 129",
    "step": "fg 129",
    "plan": "fg 39",
}


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

    def trace(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def debug(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def info(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def warning(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def error(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def critical(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102
    def exception(self, msg: str, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003, D102


class Formatter:
    """A class used to format log records.

    Attributes
    ----------
    max_lines : int
        The maximum number of lines to include in the formatted log message.

    Methods
    -------
    format(record)
        Formats a log record into a string.

    """

    def __init__(self) -> None:
        """Initialize the logger with default settings.

        Attributes:
            max_lines (int): The maximum number of lines the logger can handle, default is 30.

        """
        self.max_lines = 30

    def format(self, record: Any) -> str:  # noqa: ANN401
        """Format a log record into a string with specific formatting.

        Args:
            record (dict): A dictionary containing log record information.
                Expected keys are "message", "extra", "time", "level", "name",
                "function", and "line".

        Returns:
            str: The formatted log record string.

        """
        msg = record["message"]
        if isinstance(msg, str):
            msg = self._sanitize_message_(msg)
        function_color = self._get_function_color_(record)

        # Create the base format string
        result = (
            f"<green>{record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}</green> | "
            f"<level>{record['level'].name}</level> | "
            f"<{function_color}>{record['name']}</{function_color}>:"
            f"<{function_color}>{record['function']}</{function_color}>:"
            f"<{function_color}>{record['line']}</{function_color}> - "
            f"<level>{msg}</level>"
        )
        if record.get("exception") and hasattr(record["exception"], "value"):
            formatted_stack_trace = "".join(traceback.format_exception(record["exception"].value))
            formatted_stack_trace = self._sanitize_message_(formatted_stack_trace, truncate=False)
            result += f"\n{formatted_stack_trace}"

        # Add extra information if present
        if record["extra"]:
            result += " | {extra}"

        result += "\n"
        return result

    def _sanitize_message_(self, msg: str, truncate: bool = True) -> str:
        """Sanitize a message to be used in a log record."""
        # doubles opening curly braces in a string { -> {{
        msg = re.sub(r"(?<!\{)\{(?!\{)", "{{", msg)
        # doubles closing curly braces in a string } -> }}
        msg = re.sub(r"(?<!\})\}(?!\})", "}}", msg)
        # escapes < and > in a string
        msg = msg.replace("<", r"\<").replace(">", r"\>")

        return self._truncated_message_(msg) if truncate else msg

    def _get_function_color_(self, record: Any) -> str:  # noqa: ANN401
        """Get color based on function/module name. Default is white."""
        return next(
            (
                color
                for key, color in FUNCTION_COLOR_MAP.items()
                if any(key in field for field in [record["function"], record["name"]])
            ),
            "white",
        )

    def _truncated_message_(self, msg: str) -> str:
        lines = msg.split("\n")
        if len(lines) > self.max_lines:
            # Keep first and last parts, truncate the middle
            keep_lines = self.max_lines - 1  # Reserve one line for truncation message
            head_lines = keep_lines // 2
            tail_lines = keep_lines - head_lines

            truncated_lines = lines[:head_lines]
            truncated_lines.append(f"... (truncated {len(lines) - keep_lines} lines) ...")
            truncated_lines.extend(lines[-tail_lines:])
            msg = "\n".join(truncated_lines)
        return msg


class SafeLogger(LoggerInterface):
    """A logger that catches exceptions and logs them to the child logger."""

    def __init__(self, child_logger: LoggerInterface) -> None:
        """Initialize the SafeLogger."""
        super().__init__()
        self.child_logger = (
            child_logger.opt(depth=1)
            if isinstance(child_logger, type(default_logger))
            else child_logger
        )

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's trace method to catch exceptions."""
        try:
            self.child_logger.trace(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's debug method to catch exceptions."""
        try:
            self.child_logger.debug(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's info method to catch exceptions."""
        try:
            self.child_logger.info(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's warning method to catch exceptions."""
        try:
            self.child_logger.warning(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's error method to catch exceptions."""
        try:
            self.child_logger.error(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's exception method to catch exceptions."""
        try:
            self.child_logger.exception(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Wrap the child logger's critical method to catch exceptions."""
        try:
            self.child_logger.critical(msg, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.child_logger.error(f"Failed to log: {e}")  # noqa: G004, TRY400


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
        self.formatter = Formatter()
        default_logger.remove()
        default_logger.add(
            sys.stdout,
            level="INFO",
            format=self.formatter.format,
            serialize=False,
            catch=True,
        )
        self._logger: LoggerInterface = custom_logger or SafeLogger(default_logger)  # type: ignore  # noqa: PGH003
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
                format=self.formatter.format,
                serialize=config.json_log_serialize,
                catch=True,
            )


# Expose manager to allow updating logger
logger_manager = LoggerManager()


def logger() -> LoggerInterface:
    """Return the active logger.

    Returns:
        LoggerInterface: The current active logger being used.

    """
    return logger_manager.logger
