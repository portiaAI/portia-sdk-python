"""Logging functions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol

from loguru import logger as default_logger

if TYPE_CHECKING:
    from portia.config import Config


class LoggerInterface(Protocol):
    """General Interface for loggers."""

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
    """Manages package level logger."""

    def __init__(self, custom_logger: LoggerInterface | None = None) -> None:
        """Initialize the LoggerManager."""
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
        """Get the current logger."""
        return self._logger

    def set_logger(self, custom_logger: LoggerInterface) -> None:
        """Set a custom logger."""
        self._logger = custom_logger
        self.custom_logger = True

    def configure_from_config(self, config: Config) -> None:
        """Configure the global logger based on the library's configuration."""
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


# expose manager to allow updating logger
logger_manager = LoggerManager()


def logger() -> LoggerInterface:
    """Return active logger."""
    return logger_manager.logger
