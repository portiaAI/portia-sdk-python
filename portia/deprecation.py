"""Deprecation utilities and warnings for Portia SDK.

This module provides utilities for handling deprecation warnings and managing
feature flags for Plan V2 sunset migration.
"""

import os
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def get_plan_v2_default() -> bool:
    """Get the PLAN_V2_DEFAULT feature flag value.

    Returns:
        bool: True if PLAN_V2_DEFAULT is set to 'true', False otherwise.

    """
    return os.getenv("PLAN_V2_DEFAULT", "false").lower() == "true"


def deprecation_warning(message: str) -> None:
    """Emit a deprecation warning.

    Args:
        message: The deprecation warning message.

    """
    warnings.warn(
        message,
        DeprecationWarning,
        stacklevel=3,  # Point to the caller's caller (the actual user code)
    )


def deprecated_class(
    replacement: str,
    version: str | None = None,
) -> Callable[[type], type]:
    """Create decorator for deprecated classes.

    Args:
        replacement: The recommended replacement class.
        version: The version when the deprecation was introduced.

    Returns:
        A decorator that adds deprecation warnings to class initialization.

    """
    def decorator(cls: type) -> type:
        original_init = cls.__init__

        def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            version_info = f" (deprecated since v{version})" if version else ""
            deprecation_warning(
                f"{cls.__name__} is deprecated{version_info}. "
                f"Use {replacement} instead."
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init
        return cls
    return decorator


def deprecated_import(
    module_name: str,
    class_name: str,
    replacement: str,
    version: str | None = None,
) -> None:
    """Emit a deprecation warning for importing deprecated classes.

    Args:
        module_name: The module being imported from.
        class_name: The class being imported.
        replacement: The recommended replacement.
        version: The version when the deprecation was introduced.

    """
    version_info = f" (deprecated since v{version})" if version else ""
    deprecation_warning(
        f"Importing {class_name} from {module_name} is deprecated{version_info}. "
        f"Use {replacement} instead."
    )
