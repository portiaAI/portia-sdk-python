"""Plan primitives."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class Variable(BaseModel):
    """A variable in the plan.

    A variable is a way of referencing other parts of the plan usually either another steps output
    or a constant input variable.
    """

    model_config = ConfigDict(extra="forbid")

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


class Step(BaseModel):
    """A step in a workflow."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        description="The task that needs to be completed by this step",
    )
    inputs: list[Variable] = Field(
        default=[],
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


class ReadOnlyStep(Step):
    """A read only copy of a step, passed to agents for reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_step(cls, step: Step) -> ReadOnlyStep:
        """Configure a read only step from a normal step."""
        return cls(
            task=step.task,
            inputs=step.inputs,
            tool_name=step.tool_name,
            output=step.output,
        )


class PlanContext(BaseModel):
    """Context for a plan."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The original query given by the user.")
    tool_ids: list[str] = Field(description="The list of tools IDs available to the planner.")


class Plan(BaseModel):
    """A plan represent a series of steps that an agent should follow to execute the query."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(
        default_factory=uuid4,
        description="A unique ID for this plan.",
    )
    plan_context: PlanContext = Field(description="The context for when the plan was created.")
    steps: list[Step] = Field(description="The set of steps to solve the query.")

    @field_serializer("id")
    def serialize_id(self, plan_id: UUID) -> str:
        """Serialize the id to a string."""
        return str(plan_id)

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            f"PlanModel(id={self.id!r},"
            f"plan_context={self.plan_context!r}, "
            f"steps={self.steps!r}"
        )
