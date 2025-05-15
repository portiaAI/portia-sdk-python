"""Models for end user management."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EndUser(BaseModel):
    """Represents an actual user of the system."""

    external_id: str = Field(description="The external ID of the end user.")

    name: str = Field(default="", description="The name of the end user.")
    email: str = Field(default="", description="The email address of the end user.")
    phone_number: str = Field(default="", description="The phone number of the end user.")

    additional_data: dict[str, str | None] = Field(
        default={},
        description="Any additional data about the user.",
    )

    def set_additional_data(self, key_name: str, key_value: str) -> None:
        """Set a field in the additional data blob."""
        self.additional_data[key_name] = key_value

    def remove_additional_data(self, key_name: str) -> None:
        """Set a field in the additional data blob."""
        self.additional_data[key_name] = None

    def get_additional_data(self, key_name: str) -> str | None:
        """Get a field from the additional data blob."""
        if key_name in self.additional_data:
            return self.additional_data[key_name]
        return None
