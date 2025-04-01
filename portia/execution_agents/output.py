"""Outputs from a plan run step.

These are stored and can be used as inputs to future steps
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Generic

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from portia.common import SERIALIZABLE_TYPE_VAR, Serializable
from portia.prefixed_uuid import PlanRunUUID

if TYPE_CHECKING:
    from portia.storage import AgentMemory


class AgentMemoryStorageDetails(BaseModel):
    """Details about the storage of an output in agent memory."""

    name: str
    plan_run_id: PlanRunUUID


class Output(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Output of a tool with a wrapper for data, summaries, and LLM interpretation.

    This class contains a generic value `T` bound to `Serializable`.

    Attributes:
        value (SERIALIZABLE_TYPE_VAR | None): The output of the tool.
        summary (str | None): A textual summary of the output. Not all tools generate summaries.
          If value is stored in agent memory, this will always be set.

    """

    model_config = ConfigDict(extra="forbid")

    value: AgentMemoryStorageDetails | Serializable | None = Field(
        default=None,
        description="The output of the tool. If the value is stored in agent memory, "
        "this will contain the storage details so it can be retrieved.",
        alias="value",  # This ensures the field is serialized as "value" in JSON
    )
    summary: str | None = Field(
        default=None,
        description="Textual summary of the output of the tool. Not all tools generate summaries.",
    )

    def value_for_prompt(self) -> Serializable | None:
        """Get the value in a format suitable for an LLM prompt.

        If the output is not so large that it can't be stored locally, it will be returned. If the
        value is stored remotely in agent memory, this will be a summary of the value (i.e. the full
        value won't be fetched).

        Returns:
            Serializable | None: The value of the output.

        """
        return self.summary if self._stored_in_agent_memory() else self.value

    def full_value(self, agent_memory: AgentMemory) -> Serializable | None:
        """Get the full value, fetching from remote storage or file if necessary.

        Returns:
            Serializable | None: The full value of the output, fetched from storage if needed.

        """
        return (
            agent_memory.get_plan_run_output(self.value.name, self.value.plan_run_id)
            if self._stored_in_agent_memory()
            else self.value
        )

    def _stored_in_agent_memory(self) -> bool:
        """Whether the output is stored in agent memory."""
        return isinstance(self.value, AgentMemoryStorageDetails)

    @field_serializer("value")
    def serialize_value(self, value: Serializable | None) -> str:  # noqa: C901, PLR0911
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
