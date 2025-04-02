"""Outputs from a plan run step.

These are stored and can be used as inputs to future steps
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Generic

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from portia.common import SERIALIZABLE_TYPE_VAR, Serializable
from portia.prefixed_uuid import PlanRunUUID

if TYPE_CHECKING:
    from portia.storage import AgentMemory


class Output(ABC, BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Output of a tool with a wrapper for data, summaries, and LLM interpretation.

    Attributes:
        value (SERIALIZABLE_TYPE_VAR | None): The output of the tool.
        summary (str | None): A textual summary of the output. Not all tools generate summaries.
          If value is stored in agent memory, this will always be set.

    """

    model_config = ConfigDict(extra="forbid")

    # @@@ MOVE THIS TO SUBCLASSES + CHANGE TO METHOD
    summary: str | None = Field(
        default=None,
        description="Textual summary of the output of the tool. Not all tools generate summaries.",
    )

    @property
    def value(self) -> Serializable | None:
        """Get the value of the output.

        This will return the output in a format that is suitable for an LLM prompt. If the output
        value is so large that it isn't appropriate to include in an LLM prompt, a summary
        will be returned and you'll need to use full_value() to get the full value.
        """
        raise NotImplementedError("value is not implemented")

    @abstractmethod
    def serialize_value(self) -> str:
        """Serialize the value to a string."""
        raise NotImplementedError("serialize_value is not implemented")

    @abstractmethod
    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:
        """Get the full value, fetching from remote storage or file if necessary."""
        raise NotImplementedError("full_value is not implemented")


class LocalOutput(Output, Generic[SERIALIZABLE_TYPE_VAR]):
    """Output that is stored locally."""

    raw_value: Serializable | None = Field(
        default=None,
        description="The output of the tool.",
        alias="value",
    )

    @property
    def value(self) -> Serializable | None:
        """Get the value of the output."""
        return self.raw_value

    def serialize_value(self) -> str:
        """Serialize the value to a string."""
        return self.serialize_value_field(self.raw_value)

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:  # noqa: ARG002
        """Return the full value."""
        return self.raw_value

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

        if isinstance(value, (dict, tuple)):
            return json.dumps(value, ensure_ascii=False)  # Ensure proper JSON formatting

        if isinstance(value, set):
            return json.dumps(
                list(value),
                ensure_ascii=False,
            )  # Convert set to list before serialization

        if isinstance(value, (int, float, bool)):
            return json.dumps(value, ensure_ascii=False)  # Ensures booleans become "true"/"false"

        if isinstance(value, (datetime, date)):
            return value.isoformat()  # Convert date/time to ISO format

        if isinstance(value, Enum):
            return str(value.value)  # Convert Enums to their values

        if isinstance(value, (BaseModel)):
            return value.model_dump_json()  # Use Pydantic's built-in serialization for models

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")  # Convert bytes to string

        return str(value)  # Fallback for other types


class AgentMemoryOutput(Output):
    """Output that is stored in agent memory."""

    output_name: str
    plan_run_id: PlanRunUUID

    @property
    def value(self) -> Serializable | None:
        """Return the summary of the output as the value is too large to be retained locally."""
        return self.summary

    def serialize_value(self) -> str:
        """Serialize the value to a string."""
        # @@@ UNDO OR
        return self.summary or ""

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:
        """Get the full value, fetching from remote storage or file if necessary."""
        return agent_memory.get_plan_run_output(self.output_name, self.plan_run_id)
