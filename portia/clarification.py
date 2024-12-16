"""Clarification Primitives."""

from __future__ import annotations

from typing import Any, Generic
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_serializer, model_validator

from portia.types import SERIALIZABLE_TYPE_VAR


class Clarification(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Base Model for Clarifications.

    A Clarification represents some question that requires user input to resolve.
    For example it could be:
    - That authentication via OAuth needs to happen and the user needs to go through an OAuth flow.
    - That one argument provided for a tool is missing and the user needs to provide it.
    - That the user has given an input that is not allowed and needs to choose from a list.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="A unique ID for this clarification",
    )
    type: str = Field(
        init=False,
        repr=False,
        default="clarification",
        description="type of clarification",
    )
    response: SERIALIZABLE_TYPE_VAR | None = Field(
        default=None,
        description="The response from the user to this clarification.",
    )
    step: int | None = Field(default=None, description="The step this clarification is linked to.")
    user_guidance: str = Field(
        description="Guidance that is provided to the user to help clarification.",
    )
    resolved: bool = Field(
        default=False,
        description="Whether this clarification has been resolved.",
    )

    # LLMs can struggle to generate uuids when returning structured output
    # but as its an ID field we can assign a new ID in this case.
    @model_validator(mode="before")
    @classmethod
    def validate_uuid(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate a given uuid is valid else assign a new one."""
        uuid_value = values.get("id")
        if isinstance(uuid_value, str):
            try:
                # Try parsing the UUID string
                values["id"] = UUID(uuid_value)
            except ValueError:
                # If parsing fails, use the default_factory
                values["id"] = uuid4()
        elif not isinstance(uuid_value, UUID):
            # If missing or invalid, use the default_factory
            values["id"] = uuid4()
        return values


class ArgumentClarification(Clarification[SERIALIZABLE_TYPE_VAR]):
    """A clarification about a specific argument for a tool.

    The name of the argument should be given within the clarification.
    """

    argument_name: str


class ActionClarification(Clarification[bool]):
    """An action based clarification.

    Represents a clarification where the user needs to click on a link. Set the response to true
    once the user has clicked on the link and done the associated action.
    """

    type: str = "Action Clarification"
    action_url: HttpUrl

    @field_serializer("action_url")
    def serialize_action_url(self, action_url: HttpUrl) -> str:
        """Serialize the action URL to a string."""
        return str(action_url)


class InputClarification(ArgumentClarification[str]):
    """An input based clarification.

    Represents a clarification where the user needs to provide a value for a specific argument.
    """

    type: str = "Input Clarification"


class MultiChoiceClarification(ArgumentClarification[str]):
    """A multiple choice based clarification.

    Represents a clarification where the user needs to select an option for a specific argument.
    """

    type: str = "Multiple Choice Clarification"
    options: list[str]
