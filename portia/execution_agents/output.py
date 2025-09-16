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
    """Output data that is stored locally within the current process.

    This class represents step output data that is kept in local memory rather than
    being stored in external agent memory. It is used for step outputs that are small
    enough to be efficiently stored and accessed locally.

    Attributes:
        value: The actual data value, typically the output from a tool execution.
            Can be any serializable type (str, dict, list, BaseModel, etc.).
        summary: Optional human-readable description of the value. Not all tools
            generate summaries, and plan inputs typically don't need summaries.

    Example:
        >>> data = LocalDataValue(value="Hello World", summary="Greeting message")
        >>> data.get_value()
        "Hello World"
    """

    model_config = ConfigDict(extra="forbid")

    value: Serializable | None = Field(
        default=None,
        description="The actual data value, often the output from a tool execution. "
        "Can be any serializable type including strings, numbers, dicts, lists, or Pydantic models.",
    )

    summary: str | None = Field(
        default=None,
        description="Optional textual summary or description of the value. "
        "Provides human-readable context about what the value represents. "
        "Not all tools generate summaries and plan inputs typically don't need them.",
    )

    def get_value(self) -> Serializable | None:
        """Retrieve the stored data value.

        Returns:
            The stored value, which can be any serializable type or None.
            This is the actual data content that was stored locally.
        """
        return self.value

    def serialize_value(self) -> str:
        """Convert the stored value to its string representation.

        Returns:
            A string representation of the value suitable for storage,
            transmission, or display. Complex objects are JSON-serialized.
        """
        return self.serialize_value_field(self.value)

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:  # noqa: ARG002
        """Retrieve the complete data value.

        For LocalDataValue, since the data is stored locally, this returns
        the same value as get_value(). The agent_memory parameter is ignored
        but kept for interface compatibility.

        Args:
            agent_memory: External memory storage (unused for local values).

        Returns:
            The complete stored value, identical to get_value() for local data.
        """
        return self.value

    def get_summary(self) -> str | None:
        """Retrieve the human-readable summary of the data.

        Returns:
            The summary string if one was provided, None otherwise.
            Useful for displaying a brief description of the data content.
        """
        return self.summary

    @field_serializer("value")
    def serialize_value_field(self, value: Serializable | None) -> str:  # noqa: C901, PLR0911
        """Serialize any serializable value to its string representation.

        This method handles the conversion of various data types to strings,
        including complex objects like Pydantic models, collections, and primitives.
        It ensures proper JSON formatting for structured data.

        Args:
            value: The value to serialize. Can be any serializable type including
                str, int, float, bool, dict, list, set, tuple, BaseModel, Enum,
                datetime, date, bytes, or None.

        Returns:
            The string representation of the value. Complex objects are
            JSON-serialized, while simple types are converted appropriately.
            Returns empty string for None values.
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
    """Output data that is stored in external agent memory storage.

    This class represents step output data that is too large to be efficiently
    stored in local memory, so it's stored in external agent memory and accessed
    by reference. The actual data is retrieved on-demand using the output name
    and plan run ID.

    Attributes:
        output_name: Unique identifier for this output within the plan run.
        plan_run_id: UUID of the plan run that generated this output.
        summary: Human-readable description of the stored data content.

    Example:
        >>> memory_output = AgentMemoryValue(
        ...     output_name="large_dataset",
        ...     plan_run_id=PlanRunUUID(),
        ...     summary="Processing results for 10,000 records"
        ... )
        >>> # Access the full data later:
        >>> full_data = memory_output.full_value(agent_memory)
    """

    model_config = ConfigDict(extra="forbid")

    output_name: str = Field(
        description="Unique identifier for this output within the plan run context. "
        "Used to retrieve the actual data from agent memory storage.",
    )
    plan_run_id: PlanRunUUID = Field(
        description="UUID of the plan run that generated this output. "
        "Required to locate the data in agent memory storage.",
    )
    summary: str = Field(
        description="Human-readable textual summary describing the stored data content. "
        "This serves as the primary accessible representation since the full data "
        "is stored externally and may be large.",
    )

    def get_value(self) -> Serializable | None:
        """Retrieve the summary as a lightweight representation of the data.

        Since the actual data is stored externally and may be very large,
        this method returns the summary instead of the full value to avoid
        memory and performance issues.

        Returns:
            The summary string, which serves as a lightweight representation
            of the actual data stored in agent memory.
        """
        return self.summary

    def serialize_value(self) -> str:
        """Convert the data to its string representation.

        For AgentMemoryValue, this returns the summary since the actual data
        is stored externally and may be too large for efficient serialization.

        Returns:
            The summary string, which provides a serializable representation
            of the externally stored data.
        """
        return self.summary

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:
        """Retrieve the complete data value from external agent memory.

        This method fetches the actual stored data from agent memory using
        the output name and plan run ID. Use this when you need access to
        the full data content, not just the summary.

        Args:
            agent_memory: The agent memory storage system that contains
                the actual data for this output.

        Returns:
            The complete data value retrieved from external storage,
            or None if the data cannot be found or accessed.
        """
        return agent_memory.get_plan_run_output(self.output_name, self.plan_run_id).get_value()

    def get_summary(self) -> str:
        """Retrieve the human-readable summary of the externally stored data.

        Returns:
            The summary string that describes the data content stored
            in agent memory. This is the same value returned by get_value()
            for this class.
        """
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
