"""Types and utilities useful across the package.

This module defines various types, utilities, and base classes used throughout the package.
It includes a custom Enum class, helper functions, and base models with special configurations for
use in the Portia framework.
"""

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

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


class PortiaBaseModel(BaseModel):
    """Base model class with additional configuration.

    This class extends `BaseModel` from Pydantic and includes a custom `model_config` that forbids
    the inclusion of extra fields not defined in the model. It serves as the base class for other
    models that require such a configuration.
    """

    model_config = ConfigDict(extra="forbid")


class PortiaReadOnlyModel(BaseModel):
    """Read-only base model class.

    This class extends `BaseModel` from Pydantic and includes a custom `model_config` that freezes
    all fields, making them immutable after creation. It is intended to be used for models where the
    fields should not be modified.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
