"""Types and utilities useful across the package."""

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

Serializable = Any
SERIALIZABLE_TYPE_VAR = TypeVar("SERIALIZABLE_TYPE_VAR", bound=Serializable)


class PortiaEnum(str, Enum):
    """Base enum class for Portia enums that provides common functionality."""

    @classmethod
    def enumerate(cls) -> tuple[tuple[str, str], ...]:
        """Return a tuple of all choices as (name, value) pairs."""
        return tuple((x.name, x.value) for x in cls)


def combine_args_kwargs(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Combine Args + Kwargs into a single dict."""
    args_dict = {f"{i}": arg for i, arg in enumerate(args)}
    return {**args_dict, **kwargs}


class PortiaBaseModel(BaseModel):
    """PortiaBaseModel contains additional default model_config for our models."""

    model_config = ConfigDict(extra="forbid")


class PortiaReadOnlyModel(BaseModel):
    """PortiaReadOnlyModel is a read only version of the BaseModel that freezes fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")
