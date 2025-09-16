"""Deprecation utilities for PlanV1 sunset.

This module provides utilities for managing deprecation warnings and feature flags
related to the sunset of PlanV1 classes and methods in favor of PlanV2.
"""

import os
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from portia.logger import logger

F = TypeVar("F", bound=Callable[..., Any])

# Feature flag environment variable
PLAN_V2_DEFAULT_ENV = "PLAN_V2_DEFAULT"


def is_plan_v2_default() -> bool:
    """Check if PlanV2-first behavior is enabled via feature flag.

    Returns:
        bool: True if PLAN_V2_DEFAULT environment variable is set to 'true',
              False otherwise.

    """
    return os.getenv(PLAN_V2_DEFAULT_ENV, "false").lower() == "true"


def log_deprecation_warning(
    deprecated_item: str,
    replacement: str | None = None,
    version: str | None = None,
    stacklevel: int = 3,
) -> None:
    """Log a deprecation warning for PlanV1 classes and methods.

    This function emits both a standard Python deprecation warning and logs
    the deprecation message using the Portia logger.

    Args:
        deprecated_item: Name of the deprecated class, method, or feature
        replacement: Recommended replacement (e.g., "PlanBuilderV2")
        version: Version in which the item will be removed
        stacklevel: Stack level for the warning (default: 3)

    """
    # Build the warning message
    message_parts = [f"'{deprecated_item}' is deprecated"]

    if replacement:
        message_parts.append(f"use '{replacement}' instead")

    if version:
        message_parts.append(f"will be removed in version {version}")
    else:
        message_parts.append("and will be removed in a future version")

    message = f"{message_parts[0]} - {', '.join(message_parts[1:])}."

    # Emit Python deprecation warning
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

    # Also log using Portia's logger for visibility
    logger().warning(f"DEPRECATION: {message}")


def deprecated_class(
    replacement: str | None = None,
    version: str | None = None,
) -> Callable[[type], type]:
    """Class decorator to mark a class as deprecated.

    This decorator logs a deprecation warning when the class is instantiated.

    Args:
        replacement: Recommended replacement class name
        version: Version in which the class will be removed

    Returns:
        Decorator function that wraps the class __init__ method

    """
    def decorator(cls: type) -> type:
        original_init = cls.__init__

        @wraps(original_init)
        def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            log_deprecation_warning(
                deprecated_item=cls.__name__,
                replacement=replacement,
                version=version,
                stacklevel=4,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls

    return decorator


def deprecated_function(
    replacement: str | None = None,
    version: str | None = None,
) -> Callable[[F], F]:
    """Mark a function as deprecated.

    Args:
        replacement: Recommended replacement function name
        version: Version in which the function will be removed

    Returns:
        Decorator function that wraps the original function

    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            log_deprecation_warning(
                deprecated_item=func.__name__,
                replacement=replacement,
                version=version,
                stacklevel=3,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def warn_on_import(
    module_name: str,
    deprecated_items: list[str],
    replacement_module: str | None = None,
) -> None:
    """Emit deprecation warnings when deprecated items are imported.

    This should be called in module __init__ or at import time.

    Args:
        module_name: Name of the module containing deprecated items
        deprecated_items: List of deprecated class/function names
        replacement_module: Recommended replacement module

    """
    for item in deprecated_items:
        replacement = f"{replacement_module}.{item}" if replacement_module else None
        log_deprecation_warning(
            deprecated_item=f"{module_name}.{item}",
            replacement=replacement,
            stacklevel=4,
        )
