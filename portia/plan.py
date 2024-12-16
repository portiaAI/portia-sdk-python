"""Plan primitives."""

from __future__ import annotations

from typing import Any, Generic
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from portia.clarification import Clarification
from portia.types import SERIALIZABLE_TYPE_VAR


class Variable(BaseModel):
    """A variable in the plan.

    A variable is a way of referencing other parts of the plan usually either another steps output
    or a constant input variable.
    """

    name: str = Field(
        description=(
            "The name of the variable starting with '$'. The variable should be the output"
            " of another step, or be a constant."
        ),
    )
    value: Any = Field(
        default=None,
        description="If the value is not set, it will be defined by other preceding steps.",
    )
    description: str = Field(
        description="A description of the variable.",
    )

    def get_context_string(self) -> str:
        """Return a string representation of the variable."""
        return f"{self.name} ({self.description}): {self.value}"


class Output(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Output of a tool with wrapper for data, summaries and LLM interpretation.

    Contains a generic value T bound to Serializable and optional short and long summaries to be
    used for the contextual output of the tool and to explain result to the LLM.
    """

    value: SERIALIZABLE_TYPE_VAR | None = Field(default=None, description="The output of the tool")


class Step(BaseModel):
    """A step in a workflow."""

    task: str = Field(
        description="The task that needs to be completed by this step",
    )
    input: list[Variable] | None = Field(
        default=None,
        description=(
            "The input to the step, as a variable with name and description. "
            "Constants should also have a value. These are not the inputs to the tool "
            "necessarily, but all the inputs to the step."
        ),
    )
    tool_name: str | None = Field(
        default=None,
        description="The name of the tool listed in <Tools/>",
    )
    output: str = Field(
        ...,
        description="The unique output id of this step i.e. $best_offers.",
    )
    clarifications: list[Clarification] | None = Field(
        default=None,
        description="Clarifications for the step, if any.",
        exclude=True,
    )


class Plan(BaseModel):
    """A plan represent a series of steps that an agent should follow to execute the query."""

    id: UUID = Field(
        default_factory=uuid4,
        description="A unique ID for this plan.",
    )
    query: str = Field(description="The original query given by the user.")
    steps: list[Step] = Field(description="The set of steps to solve the query.")

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
