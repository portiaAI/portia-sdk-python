"""Context management for Portia dependency injection.

This module provides the PortiaContext class for managing dependencies
throughout the Portia system using a dependency injection pattern.
"""

from __future__ import annotations

from portia.config import Config, StorageClass
from portia.execution_hooks import ExecutionHooks
from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
from portia.telemetry.telemetry_service import BaseProductTelemetry, ProductTelemetry
from portia.tool_registry import DefaultToolRegistry, Tool, ToolRegistry


class PortiaContext:
    """Context class for dependency management in Portia.

    This class encapsulates all the dependencies that Portia needs,
    providing a centralized way to manage configuration, storage, tools,
    telemetry, and execution hooks.
    """

    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
        execution_hooks: ExecutionHooks | None = None,
        telemetry: BaseProductTelemetry | None = None,
    ) -> None:
        """Initialize the PortiaContext.

        Args:
            config: The configuration to use. If not provided, the default configuration will be used.
            tools: The registry or list of tools to use. If not provided, the default tool registry will be used.
            execution_hooks: Hooks that can be used to modify or add extra functionality to plan runs.
            telemetry: Anonymous telemetry service.

        """
        self.config = config if config else Config.from_default()
        self.execution_hooks = execution_hooks if execution_hooks else ExecutionHooks()
        self.telemetry = telemetry if telemetry else ProductTelemetry()

        # Initialize tool registry
        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, list):
            self.tool_registry = ToolRegistry(tools)
        else:
            self.tool_registry = DefaultToolRegistry(self.config)

        # Initialize storage based on config
        match self.config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=self.config.storage_dir)
            case StorageClass.CLOUD:
                self.storage = PortiaCloudStorage(config=self.config)

    @property
    def has_portia_api_key(self) -> bool:
        """Check if Portia API key is available."""
        return self.config.has_api_key("portia_api_key")
