"""Outputs from a plan run step.

These are stored and can be used as inputs to future steps
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from portia.common import Serializable
from portia.errors import StorageError


class RemoteMemoryValue(BaseModel):
    """Used for large outputs that are stored remotely in agent memory."""

    url: str


class FileMemoryValue(BaseModel):
    """Used for large outputs that are stored in agent memory in a local file."""

    path: str


class Output(BaseModel):
    """Output of a tool with a wrapper for data, summaries, and LLM interpretation.

    This class contains a generic value `T` bound to `Serializable`.

    Attributes:
        value (SERIALIZABLE_TYPE_VAR | None): The output of the tool.
        summary (str | None): A textual summary of the output. Not all tools generate summaries.
          If value is stored in agent memory, this will always be set.

    """

    model_config = ConfigDict(extra="forbid")

    _value: RemoteMemoryValue | FileMemoryValue | Serializable | None = Field(
        default=None,
        description="The output of the tool",
    )
    summary: str | None = Field(
        default=None,
        description="Textual summary of the output of the tool. Not all tools generate summaries.",
    )

    @property
    def value(self) -> Serializable | None:
        """Get the output value.

        If we have the value locally, it will be returned. If the value is stored remotely in agent
        memory this will be a summary of the value (i.e. the full value won't be fetched).

        Returns:
            Serializable | None: The value of the output.

        """
        match self._value:
            case value if isinstance(value, RemoteMemoryValue):
                return self.summary
            case value if isinstance(value, FileMemoryValue):
                return self.summary
            case _:
                return self._value

    @value.setter
    def value(self, value: RemoteMemoryValue | FileMemoryValue | Serializable | None) -> None:
        """Set the output value."""
        self._value = value

    def _fetch_remote_value(self, storage: RemoteMemoryValue) -> str:
        """Fetch a value from remote agent memory.

        Args:
            storage (RemoteStorage): The remote storage configuration.

        Returns:
            str: The fetched value.

        Raises:
            StorageError: If the fetch fails.

        """
        try:
            with httpx.Client() as client:
                response = client.get(storage.url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            raise StorageError(f"Failed to fetch remote value: {e}") from e

    def _read_file_value(self, storage: FileMemoryValue) -> str:
        """Read a value from file storage.

        Args:
            storage (FileStorage): The file storage configuration.

        Returns:
            str: The file contents.

        Raises:
            StorageError: If the file read fails.

        """
        try:
            return Path(storage.path).read_text()
        except Exception as e:
            raise StorageError(f"Failed to read file value: {e}") from e

    def full_value(self) -> Serializable | None:
        """Get the full value, fetching from remote storage or file if necessary.

        Returns:
            Serializable | None: The full value of the output, fetched from storage if needed.

        """
        match self.value:
            case value if isinstance(value, RemoteMemoryValue):
                return self._fetch_remote_value(value)
            case value if isinstance(value, FileMemoryValue):
                return self._read_file_value(value)
            case _:
                return self.value

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

        if isinstance(value, BaseModel):
            return value.model_dump_json()  # Use Pydantic's built-in serialization for models

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")  # Convert bytes to string

        return str(value)  # Fallback for other types
