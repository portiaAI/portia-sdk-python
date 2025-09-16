"""Deprecation utilities for managing PlanV1 sunset and PlanV2 migration.

This module provides utilities for logging deprecation warnings and managing feature flags
related to the transition from PlanV1 to PlanV2. It includes functionality to:

1. Log structured deprecation warnings
2. Check feature flags for PlanV2-first behavior
3. Provide migration guidance to users

Classes:
- DeprecationLogger: Manages deprecation warnings and logging

Functions:
- get_plan_v2_default_flag: Get the PLAN_V2_DEFAULT feature flag value
- warn_planv1_usage: Log deprecation warning for PlanV1 usage
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.logger import LoggerInterface


def get_plan_v2_default_flag() -> bool:
    """Get the PLAN_V2_DEFAULT feature flag value.

    This function first tries to get the flag from a loaded Config instance,
    then falls back to checking the environment variable directly.

    Returns:
        bool: True if PLAN_V2_DEFAULT is set to 'true' (case-insensitive), False otherwise.

    """
    # Note: Could check config if available, but environment variable is simpler
    # to avoid circular dependencies

    return os.getenv("PLAN_V2_DEFAULT", "false").lower() in ("true", "1", "yes")


class DeprecationLogger:
    """Logger for deprecation warnings related to PlanV1 sunset.

    This class provides structured logging for deprecation warnings, with configurable
    behavior based on feature flags and logger integration.

    Attributes:
        logger: Optional logger instance for structured logging
        use_warnings: Whether to emit Python warnings in addition to logging
        plan_v2_default: Feature flag indicating if PlanV2 should be the default

    """

    def __init__(
        self,
        logger: LoggerInterface | None = None,
        use_warnings: bool = True,
        plan_v2_default: bool | None = None,
    ) -> None:
        """Initialize the deprecation logger.

        Args:
            logger: Optional logger instance for structured logging
            use_warnings: Whether to emit Python warnings (default: True)
            plan_v2_default: Override for PLAN_V2_DEFAULT flag (default: reads from env)

        """
        self.logger = logger
        self.use_warnings = use_warnings
        self.plan_v2_default = (
            plan_v2_default if plan_v2_default is not None else get_plan_v2_default_flag()
        )

    def warn_planv1_usage(
        self,
        component_name: str,
        recommended_alternative: str,
        additional_context: str = "",
        stacklevel: int = 2,
    ) -> None:
        """Log a deprecation warning for PlanV1 component usage.

        Args:
            component_name: Name of the deprecated component (e.g., "PlanBuilder", "Plan")
            recommended_alternative: Recommended replacement (e.g., "PlanBuilderV2", "PlanV2")
            additional_context: Optional additional guidance or context
            stacklevel: Stack level for warnings (default: 2 to point to caller)

        """
        # Build the deprecation message
        base_message = (
            f"{component_name} is deprecated and will be removed in a future version. "
            f"Please migrate to {recommended_alternative}."
        )

        if additional_context:
            base_message = f"{base_message} {additional_context}"

        # Add feature flag context if enabled
        if self.plan_v2_default:
            base_message = (
                f"{base_message} Note: PLAN_V2_DEFAULT is enabled, "
                f"indicating prioritization of V2 components."
            )

        # Log using the provided logger if available
        if self.logger:
            self.logger.warning("DEPRECATION: %s", base_message)

        # Emit Python warning if requested
        if self.use_warnings:
            warnings.warn(base_message, DeprecationWarning, stacklevel=stacklevel)

    def warn_import_usage(
        self,
        module_name: str,
        imported_component: str,
        recommended_alternative: str,
        additional_context: str = "",
    ) -> None:
        """Log a deprecation warning for importing deprecated components.

        Args:
            module_name: Name of the module being imported from
            imported_component: Name of the deprecated component being imported
            recommended_alternative: Recommended replacement import
            additional_context: Optional additional guidance

        """
        context = f"Importing {imported_component} from {module_name}"
        if additional_context:
            context = f"{context}. {additional_context}"

        self.warn_planv1_usage(
            component_name=imported_component,
            recommended_alternative=recommended_alternative,
            additional_context=context,
            stacklevel=3,  # One level deeper since this is called from import time
        )


class _DeprecationLoggerSingleton:
    """Singleton class to manage global deprecation logger."""

    def __init__(self) -> None:
        """Initialize the singleton."""
        self._logger: DeprecationLogger | None = None

    def get(self) -> DeprecationLogger:
        """Get the global deprecation logger instance."""
        if self._logger is None:
            # Try to get the main logger if available
            try:
                from portia.logger import logger

                self._logger = DeprecationLogger(logger=logger())
            except ImportError:
                # Fallback to basic deprecation logger
                self._logger = DeprecationLogger()

        return self._logger

    def set(self, deprecation_logger: DeprecationLogger) -> None:
        """Set the global deprecation logger instance."""
        self._logger = deprecation_logger


# Global singleton instance
_singleton = _DeprecationLoggerSingleton()


def get_deprecation_logger() -> DeprecationLogger:
    """Get the global deprecation logger instance.

    Returns:
        DeprecationLogger: Global deprecation logger instance

    """
    return _singleton.get()


def set_deprecation_logger(deprecation_logger: DeprecationLogger) -> None:
    """Set the global deprecation logger instance.

    Args:
        deprecation_logger: The deprecation logger instance to use globally

    """
    _singleton.set(deprecation_logger)


def warn_planv1_usage(
    component_name: str,
    recommended_alternative: str,
    additional_context: str = "",
    stacklevel: int = 2,
) -> None:
    """Log PlanV1 deprecation warnings using the global logger.

    This function uses the global deprecation logger to emit warnings.

    Args:
        component_name: Name of the deprecated component
        recommended_alternative: Recommended replacement
        additional_context: Optional additional guidance
        stacklevel: Stack level for warnings

    """
    get_deprecation_logger().warn_planv1_usage(
        component_name=component_name,
        recommended_alternative=recommended_alternative,
        additional_context=additional_context,
        stacklevel=stacklevel + 1,  # Adjust for this wrapper function
    )
