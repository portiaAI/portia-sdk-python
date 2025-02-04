"""Types and utilities useful across the package.

This module defines various types, utilities, and base classes used throughout the package.
It includes a custom Enum class, helper functions, and base models with special configurations for
use in the Portia framework.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Self, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_serializer

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


class PrefixedUUID(BaseModel):
    """A UUID with an optional prefix.

    Attributes:
        prefix (str): A string prefix to prepend to the UUID. Empty by default.
        uuid (UUID): The UUID value.
        id (str): Computed property that combines the prefix and UUID.

    """

    prefix: ClassVar[str] = ""
    uuid: UUID = Field(default_factory=uuid4)

    def __str__(self) -> str:
        """Return the string representation of the PrefixedUUID.

        Returns:
            str: The prefixed UUID string.

        """
        return str(self.uuid) if self.prefix == "" else f"{self.prefix}-{self.uuid}"


    @model_serializer
    def serialize_model(self) -> str:
        """Serialize the PrefixedUUID to a string using the id property.

        Returns:
            str: The prefixed UUID string.

        """
        return str(self)


    @classmethod
    def from_string(cls, prefixed_uuid: str) -> Self:
        """Create a PrefixedUUID from a string in the format 'prefix-uuid'.

        Args:
            prefixed_uuid (str): A string in the format 'prefix-uuid'.

        Returns:
            Self: A new instance of PrefixedUUID.

        Raises:
            ValueError: If the string format is invalid or the prefix doesn't match.

        """
        if cls.prefix == "":
            return cls(uuid=UUID(prefixed_uuid))
        prefix, uuid_str = prefixed_uuid.split("-", maxsplit=1)
        if prefix != cls.prefix:
            raise ValueError(f"Prefix {prefix} does not match expected prefix {cls.prefix}")
        return cls(uuid=UUID(uuid_str))


def uuid_serializer(class_type: type[PrefixedUUID], v: str) -> PrefixedUUID:
    """Validate the ID field."""
    if isinstance(v, class_type):
        return v
    if isinstance(v, dict):
        return class_type.model_validate(v)
    return class_type.from_string(v)
