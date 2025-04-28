"""Types and utilities useful across the package.

This module defines various types, utilities, and base classes used throughout the package.
It includes a custom Enum class, helper functions, and base models with special configurations for
use in the Portia framework.
"""

from __future__ import annotations

import importlib.util
from enum import Enum
from typing import Any, TypeVar

Serializable = Any
SERIALIZABLE_TYPE_VAR = TypeVar("SERIALIZABLE_TYPE_VAR", bound=Serializable)


class PortiaEnum(str, Enum):
    """Base enum class for Portia enums.

    This class provides common functionality for Portia enums, including the ability to retrieve all
    choices as (name, value) pairs through the `enumerate` method.
    """

    @classmethod
    def enumerate(cls) -> tuple[tuple[str, str], ...]:
        """Return a tuple of all choices as (name, value) pairs.

        This method iterates through all enum members and returns their name and value in a tuple
        format.

        Returns:
            tuple: A tuple containing pairs of enum member names and values.

        """
        return tuple((x.name, x.value) for x in cls)


def combine_args_kwargs(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Combine Args + Kwargs into a single dictionary.

    This function takes arbitrary positional and keyword arguments and combines them into a single
    dictionary. Positional arguments are indexed as string keys (e.g., "0", "1", ...) while keyword
    arguments retain their names.

    Args:
        *args: Positional arguments to be included in the dictionary.
        **kwargs: Keyword arguments to be included in the dictionary.

    Returns:
        dict: A dictionary combining both positional and keyword arguments.

    """
    args_dict = {f"{i}": arg for i, arg in enumerate(args)}
    return {**args_dict, **kwargs}


EXTRAS_GROUPS_DEPENDENCIES = {
    "mistral": ["mistralai", "langchain_mistralai"],
    "mistralai": ["mistralai", "langchain_mistralai"],  # in process of refactoring
    "google": ["google.generativeai", "langchain_google_genai"],
    "ollama": ["langchain_ollama"],
    "tools-browser-local": ["browser_use"],
    "tools-browser-browserbase": ["browser_use", "browserbase"],
    "tools-pdf-reader": ["mistralai"],
}


def validate_extras_dependencies(extra_group: str, *, raise_error: bool = True) -> bool:
    """Validate that the dependencies for an extras group are installed.

    Returns True if all dependencies are installed, False otherwise.

    Args:
        extra_group (str): The extras group to validate, e.g. "mistralai" or "google".
        raise_error (bool): Whether to raise an ImportError if the dependencies are not installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise.

    """

    def are_packages_installed(packages: list[str]) -> bool:
        """Check if a list of packages are installed."""
        try:
            return all(importlib.util.find_spec(package) is not None for package in packages)
        except ImportError:
            return False

    if not are_packages_installed(EXTRAS_GROUPS_DEPENDENCIES[extra_group]):
        if raise_error:
            raise ImportError(
                f"Please install portia-sdk-python[{extra_group}] to use this functionality.",
            )
        return False
    return True
