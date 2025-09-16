"""Deprecation utilities for managing the transition from PlanV1 to PlanV2.

This module provides utilities for logging deprecation warnings and managing the
PlanV2-first behavior through feature flags.

Environment Variables:
    PLAN_V2_DEFAULT: Set to 'true' (case-insensitive) to enable additional deprecation
        warnings when importing PlanV1 classes. When enabled, import-time warnings will
        be shown for deprecated classes like Plan and PlanBuilder.

Example:
    # Enable PlanV2-first behavior with additional import warnings
    export PLAN_V2_DEFAULT=true

    # Or in Python
    import os
    os.environ["PLAN_V2_DEFAULT"] = "true"

Functions:
    get_plan_v2_default_flag(): Check if PLAN_V2_DEFAULT feature flag is enabled
    log_deprecation_warning(): Log structured deprecation warnings
    deprecated(): Decorator for marking functions/classes as deprecated
    warn_on_v1_import(): Warn when deprecated modules are imported (if flag enabled)
"""

import os
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar

from portia.logger import logger

F = TypeVar("F", bound=Callable[..., Any])


def get_plan_v2_default_flag() -> bool:
    """Get the PLAN_V2_DEFAULT feature flag value.

    Returns:
        bool: True if PLAN_V2_DEFAULT is set to 'true' (case-insensitive), False otherwise.
    """
    return os.getenv("PLAN_V2_DEFAULT", "").lower() == "true"


def log_deprecation_warning(
    deprecated_item: str,
    replacement: str,
    version: str | None = None,
    category: type[Warning] = DeprecationWarning,
) -> None:
    """Log a deprecation warning with consistent formatting.

    Args:
        deprecated_item: The deprecated class, method, or feature name.
        replacement: What users should use instead.
        version: The version when the item will be removed (optional).
        category: The warning category to use.
    """
    version_info = f" and will be removed in version {version}" if version else ""
    message = f"{deprecated_item} is deprecated{version_info}. Use {replacement} instead."

    # Log the deprecation warning
    logger().warning(f"[DEPRECATION] {message}")

    # Also emit a standard Python warning
    warnings.warn(message, category=category, stacklevel=3)


def deprecated(
    replacement: str,
    version: str | None = None,
    category: type[Warning] = DeprecationWarning,
) -> Callable[[F], F]:
    """Decorator to mark functions/classes as deprecated.

    Args:
        replacement: What users should use instead.
        version: The version when the item will be removed (optional).
        category: The warning category to use.

    Returns:
        Decorated function that logs deprecation warnings when called.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log_deprecation_warning(
                func.__name__, replacement, version, category
            )
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


def warn_on_v1_import(module_name: str, replacement: str) -> None:
    """Warn when a PlanV1-related module is imported.

    This should be called at the module level to warn users when they
    import deprecated PlanV1 modules or classes.

    Args:
        module_name: The name of the deprecated module/class being imported.
        replacement: What users should import/use instead.
    """
    if get_plan_v2_default_flag():
        log_deprecation_warning(
            f"Importing {module_name}",
            f"importing {replacement}",
            category=FutureWarning,  # Use FutureWarning to indicate breaking changes coming
        )