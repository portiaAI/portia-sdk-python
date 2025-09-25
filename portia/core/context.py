"""Core Portia context that holds all core services.

The PortiaContext is a centralized container for all core Portia services
and configuration, making it easy to pass around and access common functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.config import Config
    from portia.execution_hooks import ExecutionHooks
    from portia.logger import LoggerManager
    from portia.storage import InMemoryStorage, DiskFileStorage, PortiaCloudStorage
    from portia.telemetry.telemetry_service import BaseProductTelemetry
    from portia.tool_registry import ToolRegistry


@dataclass
class PortiaContext:
    """Central context holding all core Portia services.

    This dataclass provides a centralized container for all the core services
    and configuration that Portia needs to operate. It makes it easier to pass
    around core functionality and ensures consistent access to these services.

    Attributes:
        config: The Portia configuration settings
        logger_manager: Manager for logging functionality
        storage: Storage backend for plans and runs
        tool_registry: Registry of available tools
        telemetry: Telemetry service for tracking usage
        execution_hooks: Hooks for execution events
    """

    config: Config
    logger_manager: LoggerManager
    storage: InMemoryStorage | DiskFileStorage | PortiaCloudStorage
    tool_registry: ToolRegistry
    telemetry: BaseProductTelemetry
    execution_hooks: ExecutionHooks