"""Prefixed UUIDs.

Support for various prefixed UUIDs that append the type of UUID to the ID.
"""

from __future__ import annotations

from typing import ClassVar, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_serializer, model_validator

PLAN_UUID_PREFIX = "plan"
PLAN_RUN_UUID_PREFIX = "prun"
CLARIFICATION_UUID_PREFIX = "clar"


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

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, v: str | dict) -> dict:
        """Validate the ID field."""
        if isinstance(v, dict):
            return v
        if cls.prefix == "":
            return {
                "uuid": UUID(v),
            }
        prefix, uuid_str = v.split("-", maxsplit=1)
        if prefix != cls.prefix:
            raise ValueError(f"Prefix {prefix} does not match expected prefix {cls.prefix}")
        return {
            "uuid": UUID(uuid_str),
        }

    def __hash__(self) -> int:
        """Make PrefixedUUID hashable by using the UUID's hash.

        Returns:
            int: Hash value of the UUID.

        """
        return hash(self.uuid)


class PlanUUID(PrefixedUUID):
    """A UUID for a plan."""

    prefix: ClassVar[str] = PLAN_UUID_PREFIX


class PlanRunUUID(PrefixedUUID):
    """A UUID for a PlanRun."""

    prefix: ClassVar[str] = PLAN_RUN_UUID_PREFIX


class ClarificationUUID(PrefixedUUID):
    """A UUID for a clarification."""

    prefix: ClassVar[str] = CLARIFICATION_UUID_PREFIX
