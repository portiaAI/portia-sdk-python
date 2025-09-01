"""Milestone-based plan primitives for defining and executing runs.

This module defines milestone-based plans as an alternative to step-based plans.
A `MilestonePlan` consists of a series of milestones that represent larger goals
or checkpoints in the execution process. Each milestone can branch to different
next milestones based on criteria.

Classes in this file include:

- `Next`: Defines the next milestone to transition to based on criteria.
- `Milestone`: Represents a milestone with a task and allowed tools.
- `MilestonePlanContext`: Provides context about the milestone plan.
- `MilestonePlan`: Represents the entire milestone-based execution plan.

These classes facilitate the definition of milestone-driven runs that can
dynamically branch based on execution outcomes and criteria.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from portia.common import Serializable
from portia.prefixed_uuid import PlanUUID


class Next(BaseModel):
    """Defines the next milestone to transition to based on criteria.

    Args:
        milestone_name (str): The name of the next milestone to transition to, or "exit" to end.
        criteria (str): Natural language description of the criteria for this transition.

    """

    model_config = ConfigDict(extra="ignore")

    milestone_name: str = Field(
        description="The name of the next milestone to transition to, or 'exit' to end execution."
    )
    criteria: str = Field(
        description=(
            "Natural language description of the criteria under which this transition should occur."
        )
    )

    def pretty_print(self) -> str:
        """Return the pretty print representation of the next transition.

        Returns:
            str: A pretty print representation of the milestone name and criteria.

        """
        return f"→ {self.milestone_name}: {self.criteria}"


class Milestone(BaseModel):
    """A milestone in a MilestonePlan.

    A milestone represents a larger goal or checkpoint in the execution process.
    It contains a task description, allowed tools, and possible next transitions.

    Args:
        name (str): The unique name to identify this milestone.
        task (str): Description of what the sub-agent needs to achieve.
        allowed_tool_ids (list[str]): List of tool IDs that are allowed for this milestone.
        next (list[Next]): List of possible next milestones and their transition criteria.

    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="The unique name to identify this milestone.")
    task: str = Field(
        description="Description of what the sub-agent needs to achieve in this milestone."
    )
    allowed_tool_prefixes: list[str] = Field(
        default=[], description="List of tool IDs that are allowed for this milestone."
    )

    def pretty_print(self) -> str:
        """Return the pretty print representation of the milestone.

        Returns:
            str: A pretty print representation of the milestone's details.

        """
        tools_summary = f"{len(self.allowed_tool_prefixes)} allowed tools"
        # next_summary = (
        #     "\n      ".join([n.pretty_print() for n in self.next])
        #     if self.next
        #     else "No transitions"
        # )

        return f"- {self.name}: {self.task}\n    Tools: {tools_summary}\n"


class MilestonePlanContext(BaseModel):
    """Context for a milestone plan.

    The milestone plan context contains information about the original query and
    metadata about the milestone-based execution approach.

    Args:
        query (str): The original query given by the user.
        all_tool_ids (list[str]): All tool IDs available across all milestones.

    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The original query given by the user.")
    all_tool_ids: list[str] = Field(
        description="All tool IDs available across all milestones in the plan."
    )

    @field_serializer("all_tool_ids")
    def serialize_tool_ids(self, tool_ids: list[str]) -> list[str]:
        """Serialize the tool_ids to a sorted list.

        Returns:
            list[str]: The tool_ids as a sorted list.

        """
        return sorted(tool_ids)


class MilestonePlanInput(BaseModel):
    """An input to a milestone plan.

    Args:
        name (str): The name of the input, e.g. $api_key.
        description (str): A description of the input.

    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="The name of the input")
    description: str | None = Field(
        description=(
            "A description of the input. This is used during planning to help understand "
            "how to use the input."
        ),
        default=None,
    )
    value: Serializable | None = Field(
        description=(
            "The value of the input. This is only used when running a plan and isn't used "
            "during planning."
        ),
        default=None,
    )

    def pretty_print(self) -> str:
        """Return the pretty print representation of the milestone plan input.

        Returns:
            str: A pretty print representation of the input's name and description.

        """
        return f"{self.name}: ({self.description or 'No description'})"


class MilestonePlan(BaseModel):
    """A milestone plan represents a series of milestones that an agent should follow.

    A milestone plan defines the execution flow as a series of milestones with
    conditional transitions between them, allowing for more flexible branching
    logic compared to linear step-based plans.

    Args:
        id (PlanUUID): A unique ID for the milestone plan.
        plan_context (MilestonePlanContext): The context for when the plan was created.
        milestones (list[Milestone]): The set of milestones that make up the plan.
        plan_inputs (list[MilestonePlanInput]): The inputs required by the plan.
        starting_milestone (str): The name of the milestone to start execution with.

    """

    model_config = ConfigDict(extra="forbid")

    id: PlanUUID = Field(default_factory=PlanUUID, description="The ID of the milestone plan.")
    milestones: list[Milestone] = Field(description="The set of milestones that make up the plan.")
    plan_inputs: list[MilestonePlanInput] = Field(
        default=[], description="The inputs required by the plan."
    )
    starting_milestone: str = Field(
        description="The name of the milestone to start execution with."
    )

    def __str__(self) -> str:
        """Return the string representation of the milestone plan.

        Returns:
            str: A string representation of the plan's ID, context, and milestones.

        """
        return (
            f"MilestonePlan(id={self.id!r}, "
            f"milestones={self.milestones!r}, "
            f"inputs={self.plan_inputs!r}, "
            f"starting_milestone={self.starting_milestone!r})"
        )

    def pretty_print(self) -> str:
        """Return the pretty print representation of the milestone plan.

        Returns:
            str: A pretty print representation of the plan's details.

        """
        return "Milestones:\n" + "\n".join(
            [milestone.pretty_print() for milestone in self.milestones]
        )

    @model_validator(mode="after")
    def validate_milestone_plan(self) -> Self:
        """Validate the milestone plan.

        Checks that milestone names are unique, starting milestone exists,
        and all referenced next milestones exist.

        Returns:
            MilestonePlan: The validated milestone plan.

        """
        # Validate milestone names are unique
        milestone_names = [milestone.name for milestone in self.milestones]
        if len(milestone_names) != len(set(milestone_names)):
            raise ValueError("Milestone names must be unique")

        # Validate starting milestone exists
        if self.starting_milestone not in milestone_names:
            raise ValueError(
                f"Starting milestone '{self.starting_milestone}' does not exist in milestones"
            )

        # # Validate all referenced next milestones exist (except "exit")
        # valid_milestone_names = set(milestone_names) | {"exit"}
        # for milestone in self.milestones:
        #     for next_transition in milestone.next:
        #         if next_transition.milestone_name not in valid_milestone_names:
        #             raise ValueError(
        #                 f"Next milestone '{next_transition.milestone_name}' in milestone "
        #                 f"'{milestone.name}' does not exist. Valid options: "
        #                 f"{sorted(valid_milestone_names)}"
        #             )

        # Validate plan input names are unique
        input_names = [input_.name for input_ in self.plan_inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("Plan input names must be unique")

        return self

    def get_milestone_by_name(self, name: str) -> Milestone | None:
        """Get a milestone by its name.

        Args:
            name (str): The name of the milestone to retrieve.

        Returns:
            Milestone | None: The milestone with the given name, or None if not found.

        """
        for milestone in self.milestones:
            if milestone.name == name:
                return milestone
        return None

    @property
    def all_tool_ids(self) -> list[str]:
        """Get all tool IDs across all milestones.

        Returns:
            list[str]: All tool IDs across all milestones.

        """
        return list(
            {
                tool_id
                for milestone in self.milestones
                for tool_id in milestone.allowed_tool_prefixes
            }
        )


class MilestonePlanBuilder:
    """A builder for creating milestone plans.

    This class provides an interface for constructing milestone plans step by step.

    Example:
        plan = MilestonePlanBuilder("Find and analyze data") \
                    .milestone("data_collection", "Collect relevant data",
                              ["search_tool", "web_scraper"]) \
                    .milestone("data_analysis", "Analyze the collected data",
                              ["calculator", "chart_tool"]) \
                    .starting_milestone("data_collection") \
                    .build()

    """

    def __init__(self) -> None:
        """Initialize the builder with the plan query.

        Args:
            query (str): The original query given by the user.
            structured_output_schema (type[BaseModel] | None): The optional structured output
                schema.

        """
        self.milestones: list[Milestone] = []
        self.plan_inputs: list[MilestonePlanInput] = []
        self.starting_milestone_name: str | None = None

    def milestone(
        self, name: str, task: str, allowed_tool_prefixes: list[str] | None = None
    ) -> MilestonePlanBuilder:
        """Add a milestone to the plan.

        Args:
            name (str): The unique name to identify this milestone.
            task (str): Description of what the sub-agent needs to achieve.
            allowed_tool_ids (list[str] | None): List of allowed tool IDs for this milestone.

        Returns:
            MilestonePlanBuilder: The builder instance with the new milestone added.

        """
        if allowed_tool_prefixes is None:
            allowed_tool_prefixes = []
        allowed_tool_prefixes.append("llm_tool")

        self.milestones.append(
            Milestone(name=name, task=task, allowed_tool_prefixes=allowed_tool_prefixes)
        )
        return self

    def plan_input(self, name: str, description: str) -> MilestonePlanBuilder:
        """Add an input variable to the plan.

        Args:
            name (str): The name of the input.
            description (str): The description of the input.

        Returns:
            MilestonePlanBuilder: The builder instance with the new plan input added.

        """
        self.plan_inputs.append(MilestonePlanInput(name=name, description=description))
        return self

    def starting_milestone(self, name: str) -> MilestonePlanBuilder:
        """Set the starting milestone for the plan.

        Args:
            name (str): The name of the starting milestone.

        Returns:
            MilestonePlanBuilder: The builder instance with the starting milestone set.

        """
        self.starting_milestone_name = name
        return self

    def build(self) -> MilestonePlan:
        """Build the milestone plan.

        Returns:
            MilestonePlan: The built milestone plan.

        Raises:
            ValueError: If no starting milestone is set or no milestones exist.

        """
        if not self.milestones:
            raise ValueError("At least one milestone must be added to the plan")

        if self.starting_milestone_name is None:
            raise ValueError("Starting milestone must be set")

        return MilestonePlan(
            milestones=self.milestones,
            plan_inputs=self.plan_inputs,
            starting_milestone=self.starting_milestone_name,
        )
