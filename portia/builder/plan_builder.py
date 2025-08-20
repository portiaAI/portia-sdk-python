"""Builder for Portia plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.builder.conditionals import ConditionalBlock, ConditionalBlockClauseType
from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import default_step_name
from portia.builder.step_v2 import ConditionalStep, FunctionCall, LLMStep, SingleToolAgent, ToolRun
from portia.plan import PlanInput

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from portia.tool import Tool


class PlanBuilderError(ValueError):
    """Error in Plan definition."""


class PlanBuilderV2:
    """Builder for Portia plans."""

    def __init__(self, label: str = "Run the plan built with the Plan Builder") -> None:
        """Initialize the builder.

        Args:
            label: The label of the plan. This is used to identify the plan in the Portia dashboard.

        """
        self.plan = PlanV2(steps=[], label=label)
        self._conditional_block_stack: list[ConditionalBlock] = []

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

    @property
    def _current_conditional_branch(self) -> ConditionalBlock | None:
        """Get the current conditional branch."""
        return self._conditional_block_stack[-1] if len(self._conditional_block_stack) > 0 else None

    def if_(
        self,
        condition: Callable[..., bool] | str,
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add a step that checks a condition."""
        parent_branch = self._current_conditional_branch
        conditional_branch = ConditionalBlock(
            clause_step_indexes=[len(self.plan.steps)],
            parent_conditional_block=parent_branch,
        )
        self._conditional_block_stack.append(conditional_branch)
        self.plan.steps.append(
            ConditionalStep(
                condition=condition,
                args=args or {},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_branch=conditional_branch,
                branch_index=0,
                branch_state_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
            )
        )
        return self

    def else_if_(
        self,
        condition: Callable[..., bool],
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add a step that checks a condition."""
        if len(self._conditional_block_stack) == 0:
            raise PlanBuilderError(
                "else_if_ must be called from a conditional branch. Please add an if_ first."
            )
        self._conditional_block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=condition,
                args=args or {},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_branch=self._conditional_block_stack[-1],
                branch_index=len(self._conditional_block_stack[-1].clause_step_indexes) - 1,
                branch_state_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
            )
        )
        return self

    def else_(self) -> PlanBuilderV2:
        """Add a step that checks a condition."""
        if len(self._conditional_block_stack) == 0:
            raise PlanBuilderError(
                "else_ must be called from a conditional branch. Please add an if_ first."
            )
        self._conditional_block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=lambda: True,
                args={},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_branch=self._conditional_block_stack[-1],
                branch_index=len(self._conditional_block_stack[-1].clause_step_indexes) - 1,
                branch_state_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
            )
        )
        return self

    def endif(self) -> PlanBuilderV2:
        """Exit a conditional branch."""
        if len(self._conditional_block_stack) == 0:
            raise PlanBuilderError(
                "endif must be called from a conditional branch. Please add an if_ first."
            )
        self._conditional_block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=lambda: True,
                args={},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_branch=self._conditional_block_stack[-1],
                branch_index=len(self._conditional_block_stack[-1].clause_step_indexes) - 1,
                branch_state_type=ConditionalBlockClauseType.END_CONDITION_BLOCK,
            )
        )
        self._conditional_block_stack.pop()
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
                conditional_branch=self._current_conditional_branch,
            )
        )
        return self

    def tool_run(
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
            ToolRun(
                tool=tool,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_branch=self._current_conditional_branch,
            )
        )
        return self

    def function_call(
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
            FunctionCall(
                function=function,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_branch=self._current_conditional_branch,
            )
        )
        return self

    def single_tool_agent(
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
            SingleToolAgent(
                tool=tool,
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_branch=self._current_conditional_branch,
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
        if len(self._conditional_block_stack) > 0:
            raise PlanBuilderError(
                "endif must be called for all if_ branches. "
                "Please add an endif for all if_ branches."
            )
        return self.plan
