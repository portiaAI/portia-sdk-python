"""Outputs from a plan run step.

These are stored and can be used as inputs to future steps
"""

from __future__ import annotations

import json
from abc import abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from typing_extensions import deprecated

from portia.common import Serializable
from portia.prefixed_uuid import PlanRunUUID

if TYPE_CHECKING:
    from portia.storage import AgentMemory


class BaseOutput(BaseModel):
    """Base interface for concrete output classes to implement."""

    @abstractmethod
    def get_value(self) -> Serializable | None:
        """Return the value of the output.

        This should not be so long that it is an issue for LLM prompts.
        """

    @abstractmethod
    def serialize_value(self) -> str:
        """Serialize the value to a string."""

    @abstractmethod
    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:
        """Get the full value, fetching from remote storage or file if necessary.

        This value may be long and so is not suitable for use in LLM prompts.
        """

    @abstractmethod
    def get_summary(self) -> str | None:
        """Return the summary of the output."""


class LocalDataValue(BaseOutput):
    """Output that is stored locally."""

    model_config = ConfigDict(extra="forbid")

    value: Serializable | None = Field(
        default=None,
        description="The value, often the output from the tool",
    )

    summary: str | None = Field(
        default=None,
        description="Textual summary of the value. Note that not all tools generate summaries and "
        "plan inputs also do not need summaries.",
    )

    def get_value(self) -> Serializable | None:
        """Get the value of the output."""
        return self.value

    def serialize_value(self) -> str:
        """Serialize the value to a string."""
        return self.serialize_value_field(self.value)

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:  # noqa: ARG002
        """Return the full value.

        As the value is stored locally, this is the same as get_value() for this type of output.
        """
        return self.value

    def get_summary(self) -> str | None:
        """Return the summary of the output."""
        return self.summary

    @field_serializer("value")
    def serialize_value_field(self, value: Serializable | None) -> str:  # noqa: C901, PLR0911
        """Serialize the value to a string.

        Args:
            value (SERIALIZABLE_TYPE_VAR | None): The value to serialize.

        Returns:
            str: The serialized value as a string.

        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return json.dumps(
                [
                    item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                    for item in value
                ],
                ensure_ascii=False,
            )

        if isinstance(value, (dict | tuple)):
            return json.dumps(value, ensure_ascii=False)  # Ensure proper JSON formatting

        if isinstance(value, set):
            return json.dumps(
                list(value),
                ensure_ascii=False,
            )  # Convert set to list before serialization

        if isinstance(value, (int | float | bool)):
            return json.dumps(value, ensure_ascii=False)  # Ensures booleans become "true"/"false"

        if isinstance(value, (datetime | date)):
            return value.isoformat()  # Convert date/time to ISO format

        if isinstance(value, Enum):
            return str(value.value)  # Convert Enums to their values

        if isinstance(value, (BaseModel)):
            return value.model_dump_json()  # Use Pydantic's built-in serialization for models

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")  # Convert bytes to string

        return str(value)  # Fallback for other types


class AgentMemoryValue(BaseOutput):
    """Output that is stored in agent memory."""

    model_config = ConfigDict(extra="forbid")

    output_name: str
    plan_run_id: PlanRunUUID
    summary: str = Field(
        description="Textual summary of the output of the tool. Not all tools generate summaries.",
    )

    def get_value(self) -> Serializable | None:
        """Return the summary of the output as the value is too large to be retained locally."""
        return self.summary

    def serialize_value(self) -> str:
        """Serialize the value to a string.

        We use the summary as the value is too large to be retained locally.
        """
        return self.summary

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:
        """Get the full value, fetching from remote storage or file if necessary."""
        return agent_memory.get_plan_run_output(self.output_name, self.plan_run_id).get_value()

    def get_summary(self) -> str:
        """Return the summary of the output."""
        return self.summary


Output = LocalDataValue | AgentMemoryValue


@deprecated(
    "LocalOutput is deprecated and will be removed in the 0.4 release - "
    "use LocalDataValue instead"
)
class LocalOutput(LocalDataValue):
    """Alias of LocalDataValue kept for backwards compatibility."""


@deprecated(
    "AgentMemoryOutput is deprecated and will be removed in the 0.4 release - "
    "use AgentMemoryValue instead"
)
class AgentMemoryOutput(AgentMemoryValue):
    """Alias of AgentMemoryValue kept for backwards compatibility."""
