"""Plan primitives used to define and execute runs.

This module defines the core objects that represent the plan for executing a PlanRun.
The `Plan` class is the main structure that holds a series of steps (`Step`) to be executed by an
agent in response to a query. Each step can have inputs, an associated tool, and an output.
Variables can be used within steps to reference other parts of the plan or constants.

Classes in this file include:

- `Variable`: A variable used in the plan, referencing outputs of previous steps or constants.
- `Step`: Defines a single task that an agent will execute, including inputs and outputs.
- `ReadOnlyStep`: A read-only version of a `Step` used for passing steps to agents.
- `PlanContext`: Provides context about the plan, including the original query and available tools.
- `Plan`: Represents the entire series of steps required to execute a query.

These classes facilitate the definition of runs that can be dynamically adjusted based on the
tools, inputs, and outputs defined in the plan.

"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from portia.prefixed_uuid import PlanUUID


class PlanBuilder:
    """A builder for creating plans.

    This class provides an interface for constructing plans step by step. Requires a step to be
    added to the plan before building it.

    Example:
    plan = PlanBuilder() \
                .step("Step 1", "tool_id_1", "output_1") \
                .step("Step 2", "tool_id_2", "output_2") \
                .input("input_1", "value_1") \
                .build()

    """

    query: str
    steps: list[Step]

    def __init__(self, query: str | None = None) -> None:
        """Initialize the builder with the plan query.

        Args:
            query (str): The original query given by the user.

        """
        self.query = query if query is not None else ""
        self.steps = []

    def step(
        self,
        task: str,
        tool_id: str | None = None,
        output: str | None = None,
        inputs: list[Variable] | None = None,
        condition: str | None = None,
    ) -> PlanBuilder:
        """Add a step to the plan.

        Args:
            task (str): The task to be completed by the step.
            tool_id (str | None): The ID of the tool used in this step, if applicable.
            output (str | None): The unique output ID for the result of this step.
            inputs (list[Variable] | None): The inputs to the step
            condition (str | None): A human readable condition which controls if the step should run
              or not.

        Returns:
            PlanBuilder: The builder instance with the new step added.

        """
        if inputs is None:
            inputs = []
        if output is None:
            output = f"$output_{len(self.steps)}"
        self.steps.append(
            Step(
                task=task,
                output=output,
                inputs=inputs,
                tool_id=tool_id,
                condition=condition,
            ),
        )
        return self

    def input(
        self,
        name: str,
        description: str | None = None,
        step_index: int | None = None,
    ) -> PlanBuilder:
        """Add an input variable to the chosen step in the plan (default is the last step).

        Inputs are outputs from previous steps.

        Args:
            name (str): The name of the input.
            description (str | None): The description of the input.
            step_index (int | None): The index of the step to add the input to. If not provided,
                                    the input will be added to the last step.

        Returns:
            PlanBuilder: The builder instance with the new input added.

        """
        step_index = self._get_step_index_or_raise(step_index)
        if description is None:
            description = ""
        self.steps[step_index].inputs.append(
            Variable(name=name, description=description),
        )
        return self

    def condition(
        self,
        condition: str,
        step_index: int | None = None,
    ) -> PlanBuilder:
        """Add a condition to the chosen step in the plan (default is the last step).

        Args:
            condition (str): The condition to be added to the chosen step.
            step_index (int | None): The index of the step to add the condition to.
                If not provided, the condition will be added to the last step.

        Returns:
            PlanBuilder: The builder instance with the new condition added.

        """
        step_index = self._get_step_index_or_raise(step_index)
        self.steps[step_index].condition = condition
        return self

    def build(self) -> Plan:
        """Build the plan.

        Returns:
            Plan: The built plan.

        """
        tool_ids = list({step.tool_id for step in self.steps if step.tool_id is not None})
        return Plan(
            plan_context=PlanContext(query=self.query, tool_ids=tool_ids),
            steps=self.steps,
        )

    def _get_step_index_or_raise(self, step_index: int | None) -> int:
        """Get the index of the step to add the condition to.

        Args:
            step_index (int | None): The index of the step to add the condition to. If not provided,
                                    it will default to the last step.

        Returns:
            int: The index of the step to add the condition to.

        """
        if step_index is None:
            step_index = len(self.steps) - 1
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError("Invalid step index or no steps in the plan")
        return step_index


class Variable(BaseModel):
    """A reference to an output of a step.

    Args:
        name (str): The name of the output to reference, e.g. $best_offers.
        description (str): A description of the output.

    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        description="The name of the output to reference, e.g. $best_offers.",
    )
    description: str = Field(
        description="A description of the output.",
    )


    def pretty_print(self) -> str:
        """Return the pretty print representation of the variable.

        Returns:
            str: A pretty print representation of the variable's name, and description.

        """
        return f"{self.name}: ({self.description})"


class Step(BaseModel):
    """A step in a PlanRun.

    A step represents a task in the run to be executed. It contains inputs (variables) and
    outputs, and may reference a tool to complete the task.

    Args:
        task (str): The task that needs to be completed by this step.
        inputs (list[Vairable]): The input to the step, as an output of a previous step.
        tool_id (str | None): The ID of the tool used in this step, if applicable.
        output (str): The unique output ID for the result of this step.

    """

    model_config = ConfigDict(extra="allow")

    task: str = Field(
        description="The task that needs to be completed by this step",
    )
    inputs: list[Variable] = Field(
        default=[],
        description=("The input to the step, as a reference to an output of a previous step."),
    )
    tool_id: str | None = Field(
        default=None,
        description="The ID of the tool listed in <Tools/>",
    )
    output: str = Field(
        ...,
        description="The unique output id of this step e.g. $best_offers.",
    )
    condition: str | None = Field(
        default=None,
        description="A human readable condition which controls if the step is run or not. "
        "If provided the condition will be evaluated and the step skipped if false. "
        "The step will run by default if not provided.",
    )

    def pretty_print(self) -> str:
        """Return the pretty print representation of the step.

        Returns:
            str: A pretty print representation of the step's task, inputs, tool_id, and output.

        """
        message = (
            f"- {self.task}\n"
            f"  Inputs: {', '.join([in_variable.pretty_print() for in_variable in self.inputs])}\n"
            f"  Tool ID: {self.tool_id}\n"
            f"  Output: {self.output}\n"
        )
        if self.condition:
            message += f"  Condition: {self.condition}\n"
        return message


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
    for the planning agent to use when generating the plan.

    Args:
        query (str): The original query given by the user.
        tool_ids (list[str]): A list of tool IDs available to the planning agent.

    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The original query given by the user.")
    tool_ids: list[str] = Field(
        description="The list of tools IDs available to the planning agent.",
    )

    @field_serializer("tool_ids")
    def serialize_tool_ids(self, tool_ids: list[str]) -> list[str]:
        """Serialize the tool_ids to a sorted list.

        Returns:
            list[str]: The tool_ids as a sorted list.

        """
        return sorted(tool_ids)


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

    def pretty_print(self) -> str:
        """Return the pretty print representation of the plan.

        Returns:
            str: A pretty print representation of the plan's ID, context, and steps.

        """
        portia_tools = [tool for tool in self.plan_context.tool_ids if tool.startswith("portia:")]
        other_tools = [
            tool for tool in self.plan_context.tool_ids if not tool.startswith("portia:")
        ]
        tools_summary = f"{len(portia_tools)} portia tools, {len(other_tools)} other tools"
        return (
            f"Task: {self.plan_context.query}\n"
            f"Tools Available Summary: {tools_summary}\n"
            f"Steps:\n" + "\n".join([step.pretty_print() for step in self.steps])
        )

    @model_validator(mode="after")
    def validate_plan(self) -> Plan:
        """Validate the plan.

        Checks that all outputs + conditions are unique.

        Returns:
            Plan: The validated plan.

        """
        outputs = [step.output + (step.condition or "") for step in self.steps]
        if len(outputs) != len(set(outputs)):
            raise ValueError("Outputs + conditions must be unique")
        return self


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
