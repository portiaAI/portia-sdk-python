"""Deprecation utilities for the Portia SDK.

This module provides utilities for handling deprecation warnings in the Portia SDK,
particularly for the PlanV1 sunset. It includes:

- DeprecationLogger: A utility for logging deprecation warnings with consistent formatting
- Helper functions for common deprecation scenarios
- Integration with the existing Portia logging system

The deprecation system is designed to help users migrate from PlanV1/PlanBuilderV1 to
PlanV2/PlanBuilderV2 by providing clear warnings and migration guidance.
"""

from __future__ import annotations

import warnings
from typing import Any

from portia.logger import logger


class DeprecationLogger:
    """Centralized deprecation logging utility.

    This class provides a consistent way to log deprecation warnings across the Portia SDK,
    with special consideration for the PlanV1 to PlanV2 migration. It integrates with both
    Python's warnings system and Portia's logging system.

    Example:
        # Log a simple deprecation warning
        deprecation_logger = DeprecationLogger()
        deprecation_logger.warn(
            "PlanBuilder",
            "PlanBuilder is deprecated. Use PlanBuilderV2 instead.",
            stacklevel=2
        )

        # Log with migration guidance
        deprecation_logger.warn_with_migration(
            "PlanBuilder",
            "PlanBuilderV2",
            additional_info="See the migration guide at https://docs.portialabs.ai/migration"
        )
    """

    def __init__(self) -> None:
        """Initialize the deprecation logger."""
        self._warned_items: set[str] = set()

    def warn(
        self,
        deprecated_item: str,
        message: str,
        category: type[Warning] = DeprecationWarning,
        stacklevel: int = 2,
        once_per_session: bool = True,
    ) -> None:
        """Log a deprecation warning.

        Args:
            deprecated_item: Name of the deprecated item (class, method, etc.)
            message: Deprecation warning message
            category: Warning category to use (defaults to DeprecationWarning)
            stacklevel: Stack level for the warning (defaults to 2)
            once_per_session: If True, only warn once per deprecated item per session
        """
        if once_per_session and deprecated_item in self._warned_items:
            return

        if once_per_session:
            self._warned_items.add(deprecated_item)

        # Log to both warnings system and Portia logger
        warnings.warn(message, category=category, stacklevel=stacklevel)
        logger().warning(f"DEPRECATED: {deprecated_item} - {message}")

    def warn_with_migration(
        self,
        deprecated_item: str,
        replacement: str,
        additional_info: str | None = None,
        category: type[Warning] = DeprecationWarning,
        stacklevel: int = 2,
        once_per_session: bool = True,
    ) -> None:
        """Log a deprecation warning with migration guidance.

        Args:
            deprecated_item: Name of the deprecated item
            replacement: Name of the replacement item
            additional_info: Additional migration information
            category: Warning category to use (defaults to DeprecationWarning)
            stacklevel: Stack level for the warning (defaults to 2)
            once_per_session: If True, only warn once per deprecated item per session
        """
        message = f"{deprecated_item} is deprecated and will be removed in a future version. Use {replacement} instead."
        if additional_info:
            message += f" {additional_info}"

        self.warn(
            deprecated_item=deprecated_item,
            message=message,
            category=category,
            stacklevel=stacklevel,
            once_per_session=once_per_session,
        )

    def warn_plan_v1_usage(
        self,
        deprecated_item: str,
        stacklevel: int = 2,
        once_per_session: bool = True,
    ) -> None:
        """Log a warning specifically for PlanV1 usage.

        Args:
            deprecated_item: Name of the deprecated PlanV1 item
            stacklevel: Stack level for the warning (defaults to 2)
            once_per_session: If True, only warn once per deprecated item per session
        """
        replacement_mapping = {
            "PlanBuilder": "PlanBuilderV2",
            "Plan": "PlanV2",
            "Step": "StepV2",
        }

        replacement = replacement_mapping.get(deprecated_item, "the V2 equivalent")

        self.warn_with_migration(
            deprecated_item=deprecated_item,
            replacement=replacement,
            additional_info="See the migration guide at https://docs.portialabs.ai/plan-v2-migration",
            stacklevel=stacklevel,
            once_per_session=once_per_session,
        )

    def reset_warnings(self) -> None:
        """Reset the set of warned items.

        This allows warnings to be shown again for items that have already been warned about.
        Primarily useful for testing.
        """
        self._warned_items.clear()


# Global deprecation logger instance
deprecation_logger = DeprecationLogger()


def warn_deprecated(
    deprecated_item: str,
    message: str,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
    once_per_session: bool = True,
) -> None:
    """Convenience function for logging deprecation warnings.

    Args:
        deprecated_item: Name of the deprecated item
        message: Deprecation warning message
        category: Warning category to use (defaults to DeprecationWarning)
        stacklevel: Stack level for the warning (defaults to 2)
        once_per_session: If True, only warn once per deprecated item per session
    """
    deprecation_logger.warn(
        deprecated_item=deprecated_item,
        message=message,
        category=category,
        stacklevel=stacklevel + 1,  # Adjust for the extra function call
        once_per_session=once_per_session,
    )


def warn_plan_v1_usage(
    deprecated_item: str,
    stacklevel: int = 2,
    once_per_session: bool = True,
) -> None:
    """Convenience function for warning about PlanV1 usage.

    Args:
        deprecated_item: Name of the deprecated PlanV1 item
        stacklevel: Stack level for the warning (defaults to 2)
        once_per_session: If True, only warn once per deprecated item per session
    """
    deprecation_logger.warn_plan_v1_usage(
        deprecated_item=deprecated_item,
        stacklevel=stacklevel + 1,  # Adjust for the extra function call
        once_per_session=once_per_session,
    )


def check_and_warn_import(
    module_name: str,
    deprecated_items: dict[str, str],
    import_globals: dict[str, Any],
) -> None:
    """Check for import of deprecated items and warn accordingly.

    This function can be used in module __init__ to warn when deprecated
    items are imported.

    Args:
        module_name: Name of the module being imported from
        deprecated_items: Mapping of deprecated item names to their replacements
        import_globals: The globals() dict from the importing module
    """
    for deprecated_item, replacement in deprecated_items.items():
        if deprecated_item in import_globals:
            deprecation_logger.warn_with_migration(
                deprecated_item=f"{module_name}.{deprecated_item}",
                replacement=f"{module_name}.{replacement}",
                stacklevel=3,  # Typically called from __init__
            )