"""Shared context for Portia services.

This module defines the PortiaContext class that provides a centralized way to access
shared dependencies across different Portia services. The context pattern helps reduce
coupling and makes it easier to pass dependencies between services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.config import Config
    from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
    from portia.telemetry.telemetry_service import BaseProductTelemetry
    from portia.tool_registry import ToolRegistry


class PortiaContext:
    """Container for shared dependencies used across Portia services.

    The PortiaContext provides access to core dependencies like storage, tool registry,
    configuration, and telemetry. This pattern allows services to be decoupled from the
    main Portia class while still having access to the resources they need.

    Attributes:
        storage: The storage backend for persisting plans, runs, and users.
        tool_registry: The registry of available tools.
        config: The Portia configuration.
        telemetry: The telemetry service for tracking usage and events.

    """

    def __init__(
        self,
        storage: InMemoryStorage | DiskFileStorage | PortiaCloudStorage,
        tool_registry: ToolRegistry,
        config: Config,
        telemetry: BaseProductTelemetry,
    ) -> None:
        """Initialize the PortiaContext.

        Args:
            storage: The storage backend for persisting data.
            tool_registry: The registry of available tools.
            config: The Portia configuration.
            telemetry: The telemetry service.

        """
        self.storage = storage
        self.tool_registry = tool_registry
        self.config = config
        self.telemetry = telemetry
