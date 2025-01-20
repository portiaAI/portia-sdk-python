"""Types and utilities useful across the package."""

from enum import Enum
from typing import Any, TypeVar

Serializable = Any
SERIALIZABLE_TYPE_VAR = TypeVar("SERIALIZABLE_TYPE_VAR", bound=Serializable)

class PortiaEnum(str, Enum):
    """Base enum class for Portia enums that provides common functionality."""

    @classmethod
    def enumerate(cls) -> tuple[tuple[str, str], ...]:
        """Return a tuple of all choices as (name, value) pairs."""
        return tuple((x.name, x.value) for x in cls)
