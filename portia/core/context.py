"""PortiaContext class for holding immutable dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.config import Config
    from portia.execution_hooks import ExecutionHooks
    from portia.logger import LoggerManager
    from portia.storage import Storage
    from portia.telemetry.telemetry_service import BaseProductTelemetry
    from portia.tool_registry import ToolRegistry


@dataclass(frozen=True)
class PortiaContext:
    """Immutable context holding all core Portia dependencies.

    This class serves as a container for all the core dependencies that are
    created during Portia initialization and remain immutable throughout the
    execution lifecycle. This provides the foundation for service decomposition
    while maintaining all existing functionality.

    Attributes:
        config: The Portia configuration object.
        logger_manager: The logger manager for configuring logging.
        storage: The storage backend (memory, disk, or cloud).
        tool_registry: Registry of available tools.
        telemetry: Telemetry service for analytics.
        execution_hooks: Hooks for customizing execution behavior.
    """

    config: Config
    logger_manager: LoggerManager
    storage: Storage
    tool_registry: ToolRegistry
    telemetry: BaseProductTelemetry
    execution_hooks: ExecutionHooks

    @classmethod
    def from_portia_init_params(
        cls,
        config: Config,
        logger_manager: LoggerManager,
        storage: Storage,
        tool_registry: ToolRegistry,
        telemetry: BaseProductTelemetry,
        execution_hooks: ExecutionHooks,
    ) -> PortiaContext:
        """Create PortiaContext from Portia initialization parameters.

        This factory method creates a PortiaContext from the parameters
        that are currently initialized in Portia.__init__.

        Args:
            config: The configuration object.
            logger_manager: The logger manager.
            storage: The storage backend.
            tool_registry: The tool registry.
            telemetry: The telemetry service.
            execution_hooks: The execution hooks.

        Returns:
            A new PortiaContext instance.
        """
        return cls(
            config=config,
            logger_manager=logger_manager,
            storage=storage,
            tool_registry=tool_registry,
            telemetry=telemetry,
            execution_hooks=execution_hooks,
        )