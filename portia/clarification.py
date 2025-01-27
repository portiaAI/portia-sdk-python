"""Clarification Primitives."""

from __future__ import annotations

from typing import Generic, Self, Union
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_serializer,
    model_validator,
)

from portia.common import SERIALIZABLE_TYPE_VAR, PortiaEnum


class ClarificationCategory(PortiaEnum):
    """The category of a clarification."""

    ARGUMENT = "Argument"
    ACTION = "Action"
    BASE = "Base"
    INPUT = "Input"
    MULTIPLE_CHOICE = "Multiple Choice"
    VALUE_CONFIRMATION = "Value Confirmation"


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
    category: ClarificationCategory = Field(
        default=ClarificationCategory.BASE,
        description="The category of this clarification",
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


class ArgumentClarification(Clarification[SERIALIZABLE_TYPE_VAR]):
    """A clarification about a specific argument for a tool.

    The name of the argument should be given within the clarification.
    """

    argument_name: str
    category: ClarificationCategory = Field(
        default=ClarificationCategory.ARGUMENT,
        description="The category of this clarification",
    )


class ActionClarification(Clarification[SERIALIZABLE_TYPE_VAR]):
    """An action based clarification.

    Represents a clarification where the user needs to click on a link. Set the response to true
    once the user has clicked on the link and done the associated action.
    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.ACTION,
        description="The category of this clarification",
    )
    action_url: HttpUrl

    @field_serializer("action_url")
    def serialize_action_url(self, action_url: HttpUrl) -> str:
        """Serialize the action URL to a string."""
        return str(action_url)


class InputClarification(ArgumentClarification[SERIALIZABLE_TYPE_VAR]):
    """An input based clarification.

    Represents a clarification where the user needs to provide a value for a specific argument.
    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.INPUT,
        description="The category of this clarification",
    )


class MultipleChoiceClarification(ArgumentClarification[SERIALIZABLE_TYPE_VAR]):
    """A multiple choice based clarification.

    Represents a clarification where the user needs to select an option for a specific argument.
    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.MULTIPLE_CHOICE,
        description="The category of this clarification",
    )
    options: list[SERIALIZABLE_TYPE_VAR]

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        """Ensure provided response is an option."""
        if self.resolved and self.response not in self.options:
            raise ValueError(f"{self.response} is not a supported option")
        return self


class ValueConfirmationClarification(ArgumentClarification[SERIALIZABLE_TYPE_VAR]):
    """A value confirmation clarification.

    Represents a clarification where the user is presented a value and needs to accept it.
    The clarification should be created with the response field already set. The user will
    denote acceptance by setting the resolved flag.
    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.VALUE_CONFIRMATION,
        description="The category of this clarification",
    )

ClarificationType = Union[
    Clarification,
    InputClarification,
    ActionClarification,
    MultipleChoiceClarification,
    ValueConfirmationClarification,
]

ClarificationListType = list[ClarificationType]
