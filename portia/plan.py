"""Plan primitives used to define and execute workflows.

This module defines the core objects that represent the plan for executing a workflow.
The `Plan` class is the main structure that holds a series of steps (`Step`) to be executed by an
agent in response to a query. Each step can have inputs, an associated tool, and an output.
Variables can be used within steps to reference other parts of the plan or constants.

Classes in this file include:

- `Variable`: A variable used in the plan, referencing outputs of previous steps or constants.
- `Step`: Defines a single task that an agent will execute, including inputs and outputs.
- `ReadOnlyStep`: A read-only version of a `Step` used for passing steps to agents.
- `PlanContext`: Provides context about the plan, including the original query and available tools.
- `Plan`: Represents the entire series of steps required to execute a query.

These classes facilitate the definition of workflows that can be dynamically adjusted based on the
tools, inputs, and outputs defined in the plan.

"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portia.prefixed_uuid import PlanUUID


class PlanBuilder:
    """A builder for creating plans.

    This class provides an interface for constructing plans step by step.

    """

    query: str
    tool_ids: list[str]
    steps: list[Step]

    def __init__(self, query: str) -> None:
        """Initialize the builder with the query and tool IDs.

        Args:
            query (str): The original query given by the user.
            tool_ids (list[str]): A list of tool IDs available to the planner.
            variables (list[Variable]): A list of variables available to the planner.
            steps (list[Step]): A list of steps available to the planner.

        """
        self.query = query
        self.tool_ids = []
        self.steps = []

    def step(
        self,
        task: str,
        tool_id: str | None = None,
        output: str | None = None,
        inputs: list[Variable] | None = None,
    ) -> PlanBuilder:
        """Add a step to the plan.

        Args:
            task (str): The task to be completed by the step.
            tool_id (str | None): The ID of the tool used in this step, if applicable.
            output (str | None): The unique output ID for the result of this step.
            inputs (list[Variable] | None): The inputs to the step

        Returns:
            PlanBuilder: The builder instance with the new step added.

        """
        if inputs is None:
            inputs = []
        if output is None:
            output = f"$output_{len(self.steps)}"
        if tool_id is not None and tool_id not in self.tool_ids:
            self.tool_ids.append(tool_id)
        self.steps.append(Step(task=task, output=output, inputs=inputs, tool_id=tool_id))
        return self

    def input(
        self,
        name: str,
        value: Any | None = None,
        description: str | None = None,  # noqa: ANN401
    ) -> PlanBuilder:
        """Add an input variable to the last step in the plan.

        Args:
            name (str): The name of the input.
            value (Any | None): The value of the input.
            description (str | None): The description of the input.

        Returns:
            PlanBuilder: The builder instance with the new input added.

        """
        if len(self.steps) == 0:
            raise ValueError("No steps in the plan")
        if description is None:
            description = ""
        self.steps[-1].inputs.append(Variable(name=name, value=value, description=description))
        return self

    def build(self) -> Plan:
        """Build the plan.

        Returns:
            Plan: The built plan.

        """
        return Plan(
            id=PlanUUID(),
            plan_context=PlanContext(query=self.query, tool_ids=self.tool_ids),
            steps=self.steps,
        )


class Variable(BaseModel):
    """A variable in the plan.

    A variable is a way of referencing other parts of the plan, usually either another step's output
    or a constant input variable.

    Args:
        name (str): The name of the variable starting with '$'. The variable should be the output
                    of another step, or be a constant.
        value (Any): The value of the variable, which may be set by other preceding steps if not
                     defined.
        description (str): A description of the variable.

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
    """A step in a workflow.

    A step represents a task in the workflow to be executed. It contains inputs (variables) and
    outputs, and may reference a tool to complete the task.

    Args:
        task (str): The task that needs to be completed by this step.
        inputs (list[Variable]): The input to the step, which can include constants and variables.
        tool_id (str | None): The ID of the tool used in this step, if applicable.
        output (str): The unique output ID for the result of this step.

    """

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
    tool_id: str | None = Field(
        default=None,
        description="The ID of the tool listed in <Tools/>",
    )
    output: str = Field(
        ...,
        description="The unique output id of this step i.e. $best_offers.",
    )


class ReadOnlyStep(Step):
    """A read-only copy of a step, passed to agents for reference.

    This class creates an immutable representation of a step, which is used to ensure agents
    do not modify the original plan during execution.

    Args:
        step (Step): A step object from which to create a read-only version.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_step(cls, step: Step) -> ReadOnlyStep:
        """Create a read-only step from a normal step.

        Args:
            step (Step): The step to be converted to read-only.

        Returns:
            ReadOnlyStep: A new read-only step.

        """
        return cls(
            task=step.task,
            inputs=step.inputs,
            tool_id=step.tool_id,
            output=step.output,
        )


class PlanContext(BaseModel):
    """Context for a plan.

    The plan context contains information about the original query and the tools available
    for the planner to use when generating the plan.

    Args:
        query (str): The original query given by the user.
        tool_ids (list[str]): A list of tool IDs available to the planner.

    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The original query given by the user.")
    tool_ids: list[str] = Field(description="The list of tools IDs available to the planner.")


class Plan(BaseModel):
    """A plan represents a series of steps that an agent should follow to execute the query.

    A plan defines the entire sequence of steps required to process a query and generate a result.
    It also includes the context in which the plan was created.

    Args:
        id (PlanUUID): A unique ID for the plan.
        plan_context (PlanContext): The context for when the plan was created.
        steps (list[Step]): The set of steps that make up the plan.

    """

    model_config = ConfigDict(extra="forbid")

    id: PlanUUID = Field(
        default_factory=PlanUUID,
        description="The ID of the plan.",
    )
    plan_context: PlanContext = Field(description="The context for when the plan was created.")
    steps: list[Step] = Field(description="The set of steps to solve the query.")

    def __str__(self) -> str:
        """Return the string representation of the plan.

        Returns:
            str: A string representation of the plan's ID, context, and steps.

        """
        return (
            f"PlanModel(id={self.id!r},"
            f"plan_context={self.plan_context!r}, "
            f"steps={self.steps!r}"
        )


class ReadOnlyPlan(Plan):
    """A read-only copy of a plan, passed to agents for reference.

    This class provides a non-modifiable view of a plan instance,
    ensuring that agents can access plan details without altering them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_plan(cls, plan: Plan) -> ReadOnlyPlan:
        """Create a read-only plan from a normal plan.

        Args:
            plan (Plan): The original plan instance to create a read-only copy from.

        Returns:
            ReadOnlyPlan: A new read-only instance of the provided plan.

        """
        return cls(
            id=plan.id,
            plan_context=plan.plan_context,
            steps=plan.steps,
        )
