"""Builder for Portia plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import default_step_name
from portia.builder.step_v2 import FunctionStep, InvokeToolStep, LLMStep, SingleToolAgentStep
from portia.plan import PlanInput

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from portia.tool import Tool


class PlanBuilderV2:
    """Builder for Portia plans."""

    def __init__(self, label: str = "Run the plan built with the Plan Builder") -> None:
        """Initialize the builder.

        Args:
            label: The label of the plan. This is used to identify the plan in the Portia dashboard.

        """
        self.plan = PlanV2(steps=[], label=label)

    def input(
        self,
        *,
        name: str,
        description: str | None = None,
        default_value: Any | None = None,  # noqa: ANN401
    ) -> PlanBuilderV2:
        """Add an input to the plan.

        Args:
            name: The name of the input.
            description: The description of the input.
            default_value: The default value of the input.

        """
        self.plan.plan_inputs.append(
            PlanInput(name=name, description=description, value=default_value)
        )
        return self

    def llm_step(
        self,
        *,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that sends a query to the underlying LLM.

        Args:
            task: The task to perform.
            inputs: The inputs to the task. The inputs can be references to previous step outputs /
              plan inputs (using StepOutput / Input) or just plain values. They are passed in as
              additional context to the LLM when it is completing the task.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            LLMStep(
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def invoke_tool_step(
        self,
        *,
        tool: str | Tool,
        args: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that directly invokes a tool.

        Args:
            tool: The tool to invoke. Should either be the id of the tool to call, the Tool instance
              to call, or a python function that should be called.
            args: The arguments to the tool. If any of these values are instances of StepOutput or
              Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            InvokeToolStep(
                tool=tool,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def function_step(
        self,
        *,
        function: Callable[..., Any],
        args: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that directly invokes a function.

        Args:
            function: The function to invoke.
            args: The arguments to the function. If any of these values are instances of StepOutput
              or Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            FunctionStep(
                function=function,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def single_tool_agent_step(
        self,
        *,
        tool: str,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that uses the execution agent with a tool.

        Args:
            tool: The tool to use.
            task: The task to perform.
            inputs: The inputs to the task. If any of these values are instances of StepOutput or
              Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            SingleToolAgentStep(
                tool=tool,
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def final_output(
        self,
        output_schema: type[BaseModel] | None = None,
        summarize: bool = False,
    ) -> PlanBuilderV2:
        """Set the final output of the plan.

        Args:
            output_schema: The schema for the final output. If provided, an LLM will be used to
              coerce the output to this schema.
            summarize: Whether to summarize the final output. If True, a summary of the final output
              will be provided along with the value.

        """
        self.plan.final_output_schema = output_schema
        self.plan.summarize = summarize
        return self

    def build(self) -> PlanV2:
        """Return the plan, ready to run."""
        return self.plan
