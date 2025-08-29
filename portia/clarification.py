"""Clarification Primitives.

This module defines base classes and utilities for handling clarifications in the Portia system.
Clarifications represent questions or actions requiring user input to resolve, with different types
of clarifications for various use cases such as arguments, actions, inputs, multiple choices,
and value confirmations.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Self

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_serializer,
    field_validator,
    model_validator,
)

from portia.common import PortiaEnum, Serializable
from portia.prefixed_uuid import ClarificationUUID, PlanRunUUID


class ClarificationCategory(PortiaEnum):
    """The category of a clarification.

    This enum defines the different categories of clarifications that can exist, such as arguments,
    actions, inputs, and more. It helps to categorize clarifications for easier
    handling and processing.
    """

    ACTION = "Action"
    INPUT = "Input"
    MULTIPLE_CHOICE = "Multiple Choice"
    VALUE_CONFIRMATION = "Value Confirmation"
    USER_VERIFICATION = "User Verification"
    CUSTOM = "Custom"


class Clarification(BaseModel, ABC):
    """Base Model for Clarifications.

    A Clarification represents a question or action that requires user input to resolve. For example
    it could indicate the need for OAuth authentication, missing arguments for a tool
    or a user choice from a list.

    Attributes:
        id (ClarificationUUID): A unique identifier for this clarification.
        category (ClarificationCategory): The category of this clarification, indicating its type.
        response (SERIALIZABLE_TYPE_VAR | None): The user's response to this clarification, if any.
        step (int | None): The step this clarification is associated with, if applicable.
        user_guidance (str): Guidance provided to the user to assist with the clarification.
        resolved (bool): Whether the clarification has been resolved by the user.

    """

    id: ClarificationUUID = Field(
        default_factory=ClarificationUUID,
        description="A unique ID for this clarification",
    )
    plan_run_id: PlanRunUUID | None = Field(
        default=None,
        description="The run this clarification is for",
    )
    category: ClarificationCategory = Field(
        description="The category of this clarification",
    )
    response: Serializable | None = Field(
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
    source: str | None = Field(
        default=None,
        description="The source of the clarification. This should be a string that identifies the "
        "origin of the clarification, such as a tool name or agent name.",
    )


class ActionClarification(Clarification):
    """Action-based clarification.

    Represents a clarification that involves an action, such as clicking a link. The response is set
    to `True` once the user has completed the action associated with the link.

    Attributes:
        category (ClarificationCategory): The category for this clarification, 'Action'.
        action_url (HttpUrl): The URL for the action that the user needs to complete.
        require_confirmation (bool): Whether the user needs to confirm once the action has been
            completed.

    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.ACTION,
        description="The category of this clarification",
    )
    action_url: HttpUrl
    require_confirmation: bool = Field(
        default=False,
        description="Whether the user needs to confirm once the action has been completed.",
    )

    @field_serializer("action_url")
    def serialize_action_url(self, action_url: HttpUrl) -> str:
        """Serialize the action URL to a string.

        Args:
            action_url (HttpUrl): The URL to be serialized.

        Returns:
            str: The serialized string representation of the URL.

        """
        return str(action_url)


class InputClarification(Clarification):
    """Input-based clarification.

    Represents a clarification where the user needs to provide a value for a specific argument.
    This type of clarification is used when the user is prompted to enter a value.

    Attributes:
        category (ClarificationCategory): The category for this clarification, 'Input'.

    """

    argument_name: str = Field(
        description="The name of the argument that a value is needed for.",
    )
    category: ClarificationCategory = Field(
        default=ClarificationCategory.INPUT,
        description="The category of this clarification",
    )


class MultipleChoiceClarification(Clarification):
    """Multiple choice-based clarification.

    Represents a clarification where the user needs to select an option for a specific argument.
    The available options are provided, and the user must select one.

    Attributes:
        category (ClarificationCategory): The category for this clarification 'Multiple Choice'.
        options (list[Serializable]): The available options for the user to choose from.

    Methods:
        validate_response: Ensures that the user's response is one of the available options.

    """

    argument_name: str = Field(
        description="The name of the argument that a value is needed for.",
    )
    category: ClarificationCategory = Field(
        default=ClarificationCategory.MULTIPLE_CHOICE,
        description="The category of this clarification",
    )
    options: list[Serializable]

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        """Ensure the provided response is an option.

        This method checks that the response provided by the user is one of the options. If not,
        it raises an error.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If the response is not one of the available options.

        """
        if self.resolved and self.response not in self.options:
            raise ValueError(f"{self.response} is not a supported option")
        return self


class ValueConfirmationClarification(Clarification):
    """Value confirmation clarification.

    Represents a clarification where the user is presented with a value and must confirm or deny it.
    The clarification should be created with the response field already set, and the user indicates
    acceptance by setting the resolved flag to `True`.

    Attributes:
        category (ClarificationCategory): The category for this clarification, 'Value Confirmation'.

    """

    argument_name: str = Field(
        description="The name of the argument that whose value needs confirmation.",
    )
    category: ClarificationCategory = Field(
        default=ClarificationCategory.VALUE_CONFIRMATION,
        description="The category of this clarification",
    )


class UserVerificationClarification(Clarification):
    """User verification clarification.

    Represents a clarification where the user some information that they must verify.

    Attributes:
        category (ClarificationCategory): The category for this clarification, 'User Verification'.

    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.USER_VERIFICATION,
        description="The category of this clarification",
    )

    @field_validator("response")
    @classmethod
    def validate_response(cls, v: Any) -> Any:  # noqa: ANN401
        """Validate that response is a boolean value or None.

        Args:
            v: The value to validate.

        Returns:
            Any: The validated value.

        Raises:
            ValueError: If the response is not a boolean.

        """
        if v is not None and not isinstance(v, bool):
            raise ValueError("response must be a boolean value or None")
        return v

    @property
    def user_confirmed(self) -> bool:
        """Whether the user has confirmed the verification.

        Returns the response attribute as a boolean value.
        """
        return bool(self.response)


class CustomClarification(Clarification):
    """Custom clarifications.

    Allows the user to extend clarifications with arbitrary data.
    The user is responsible for handling this clarification type.

    Attributes:
        category (ClarificationCategory): The category for this clarification, 'Custom'.

    """

    category: ClarificationCategory = Field(
        default=ClarificationCategory.CUSTOM,
        description="The category of this clarification",
    )
    name: str = Field(
        description="The name of this clarification."
        "Used to differentiate between different types of custom clarifications.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data for this clarification. Can include any serializable type.",
    )


"""Type that encompasses all possible clarification types."""
ClarificationType = (
    Clarification
    | InputClarification
    | ActionClarification
    | MultipleChoiceClarification
    | ValueConfirmationClarification
    | UserVerificationClarification
    | CustomClarification
)


"""A list of clarifications of any type."""
ClarificationListType = list[ClarificationType]
