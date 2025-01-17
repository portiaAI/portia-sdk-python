"""Clarification Primitives."""

from __future__ import annotations

from typing import Generic
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_serializer,
    model_validator,
)

from portia.common import SERIALIZABLE_TYPE_VAR


class Clarification(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Base Model for Clarifications.

    A Clarification represents some question that requires user input to resolve.
    For example it could be:
    - That authentication via OAuth needs to happen and the user needs to go through an OAuth flow.
    - That one argument provided for a tool is missing and the user needs to provide it.
    - That the user has given an input that is not allowed and needs to choose from a list.
    """

    model_config = ConfigDict(extra="forbid")

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

    def resolve(self, response: SERIALIZABLE_TYPE_VAR | None) -> None:
        """Resolve the clarification with the given response."""
        self.response = response
        self.resolved = True


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


class InputClarification(ArgumentClarification[SERIALIZABLE_TYPE_VAR]):
    """An input based clarification.

    Represents a clarification where the user needs to provide a value for a specific argument.
    """

    type: str = "Input Clarification"


class MultiChoiceClarification(ArgumentClarification[SERIALIZABLE_TYPE_VAR]):
    """A multiple choice based clarification.

    Represents a clarification where the user needs to select an option for a specific argument.
    """

    type: str = "Multiple Choice Clarification"
    options: list[str]

    @model_validator(mode="after")
    def validate_response(
        self,
    ) -> MultiChoiceClarification:
        """Ensure provided response is an option."""
        if self.response not in self.options:
            raise ValueError(f"{self.response} is not a supported option")
        return self


class ValueConfirmationClarification(ArgumentClarification[SERIALIZABLE_TYPE_VAR]):
    """A value acceptance clarification.

    Represents a clarification where the user is presented a value and needs to accept it.
    The clarification should be created with the response field already set. The user will
    denote acceptance by setting the resolved flag.
    """

    type: str = "Value Acceptance Clarification"

    def resolve(self, response: str | None) -> None:  # noqa: ARG002
        """Resolve the clarification but don't update the response."""
        self.resolved = True
