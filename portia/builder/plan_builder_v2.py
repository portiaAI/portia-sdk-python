"""Builder class for constructing :class:`PlanV2` instances.

You can view an example of this class in use in example_builder.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from portia.builder.conditionals import ConditionalBlock, ConditionalBlockClauseType
from portia.builder.loops import LoopBlock, LoopStepType, LoopType
from portia.builder.plan_v2 import PlanV2
from portia.builder.reference import Reference, default_step_name
from portia.builder.step_v2 import (
    ConditionalStep,
    InvokeToolStep,
    LLMStep,
    LoopStep,
    ReActAgentStep,
    SingleToolAgentStep,
    StepV2,
    UserInputStep,
    UserVerifyStep,
)
from portia.plan import PlanInput
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.telemetry.views import PlanV2BuildTelemetryEvent
from portia.tool_decorator import tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pydantic import BaseModel

    from portia.common import Serializable
    from portia.tool import Tool


class PlanBuilderError(ValueError):
    """Error in Plan definition."""


class PlanBuilderV2:
    """Chainable builder used to assemble Portia plans.

    See example_builder.py for a complete example of how to use this class.
    """

    def __init__(self, label: str = "Run the plan built with the Plan Builder") -> None:
        """Initialize the builder.

        Args:
            label: Human readable label for the plan shown in the Portia dashboard.

        """
        self.plan = PlanV2(steps=[], label=label)
        self._block_stack: list[ConditionalBlock | LoopBlock] = []

    def input(
        self,
        *,
        name: str,
        description: str | None = None,
        default_value: Any | None = None,  # noqa: ANN401
    ) -> PlanBuilderV2:
        """Add an input required by the plan.

        Inputs are values that are provided when running the plan, rather than when building the
        plan. You specify them with this .input() method when building the plan, and you can then
        use the value in the plan by using Input() references in your plan steps. Then, when you run
        the plan, you can pass in the value for the input and the references will all be substituted
        with the value.

        Args:
            name: Name of the input.
            description: Optional description for the input. This can be useful when describing what
              the input is used for, both to human users and when passing the input into language
              models.
            default_value: Optional default value to be used if no value is provided when running
              the plan.

        """
        self.plan.plan_inputs.append(
            PlanInput(name=name, description=description, value=default_value)
        )
        return self

    @property
    def _current_conditional_block(self) -> ConditionalBlock | None:
        """Get the current conditional block."""
        if len(self._block_stack) == 0:
            return None
        last_block = self._block_stack[-1]
        return last_block if isinstance(last_block, ConditionalBlock) else None

    @property
    def _current_loop_block(self) -> LoopBlock | None:
        """Get the current loop block."""
        if len(self._block_stack) == 0:
            return None
        last_block = self._block_stack[-1]
        return last_block if isinstance(last_block, LoopBlock) else None

    def loop(
        self,
        while_: Callable[..., bool] | str | None = None,
        do_while_: Callable[..., bool] | str | None = None,
        over: Reference | Sequence[Any] | None = None,
        args: dict[str, Any] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Start a new loop block.

        This creates a loop that can iterate over a sequence of values or repeat while a condition
        is true. You must specify exactly one of the loop types: while_, do_while_, or over.

        For 'while' loops, the condition is checked before each iteration:
        PlanBuilderV2()
            .loop(while_=lambda: some_condition)
            .llm_step(task="Repeats while true")
            .end_loop()

        For 'do_while' loops, the condition is checked after each iteration:
        PlanBuilderV2()
            .loop(do_while_=lambda: some_condition)
            .llm_step(task="Runs at least once")
            .end_loop()

        For 'for_each' loops, iterate over a sequence or reference:
        PlanBuilderV2().loop(over=Input("items")).llm_step(task="Process each item").end_loop()

        After opening a loop block with this loop method, you must close the block with
        .end_loop() later in the plan.

        Args:
            while_: Condition function or string to check before each iteration. The loop continues
                while this condition evaluates to True.
            do_while_: Condition function or string to check after each iteration. The loop
                continues while this condition evaluates to True, but always runs at least once.
            over: Reference or sequence to iterate over. Each iteration will process one item
                from this sequence.
            args: Arguments passed to condition functions if they are functions. These are unused
                if the condition is a string.
            step_name: Optional explicit name for the step. This allows its output to be referenced
                via StepOutput("name_of_step") rather than by index.

        """
        # Validate that exactly one of while_, do_while_, or over is set
        loop_params = [while_, do_while_, over]
        set_params = [param for param in loop_params if param is not None]

        if len(set_params) == 0:
            raise PlanBuilderError("Exactly one of while_, do_while_, or over must be set.")
        if len(set_params) > 1:
            raise PlanBuilderError("Only one of while_, do_while_, or over can be set.")

        # Determine loop type and condition
        if over is not None:
            loop_type = LoopType.FOR_EACH
            condition = None
        elif while_ is not None:
            loop_type = LoopType.WHILE
            condition = while_
        else:  # do_while_ is not None
            loop_type = LoopType.DO_WHILE
            condition = do_while_

        loop_block = LoopBlock(
            start_step_index=len(self.plan.steps),
            end_step_index=None,
        )
        self._block_stack.append(loop_block)
        self.plan.steps.append(
            LoopStep(
                step_name=step_name or default_step_name(len(self.plan.steps)),
                over=over,
                condition=condition,
                args=args or {},
                loop_block=loop_block,
                loop_step_type=LoopStepType.START,
                loop_type=loop_type,
            )
        )
        return self

    def end_loop(self, step_name: str | None = None) -> PlanBuilderV2:
        """Close the most recently opened loop block.

        This method must be called to properly close any loop block that was opened with
        loop(). It marks the end of the loop logic and allows the plan to continue with
        steps outside the loop. For example:

        # Simple while loop
        PlanBuilderV2().loop(while_=lambda: some_condition).llm_step(task="Repeats").end_loop()
        .llm_step(task="Runs after loop")

        # For-each loop over input items
        PlanBuilderV2().loop(over=Input("items")).llm_step(task="Process item").end_loop()
        .llm_step(task="Runs after all items processed")

        Failing to call end_loop() after opening a loop block with loop() will result in an
        error when building the plan.

        Args:
            step_name: Optional explicit name for the step. This allows its output to be referenced
                via StepOutput("name_of_step") rather than by index.

        """
        if len(self._block_stack) == 0 or not (loop_block := self._current_loop_block):
            raise PlanBuilderError(
                "endloop must be called from a loop block. Please add a loop first."
            )
        loop_block.end_step_index = len(self.plan.steps)
        start_loop_step = self.plan.steps[loop_block.start_step_index]
        if not isinstance(start_loop_step, LoopStep):
            raise PlanBuilderError("The step at the start of the loop is not a LoopStep")
        self.plan.steps.append(
            LoopStep(
                step_name=step_name or default_step_name(len(self.plan.steps)),
                condition=start_loop_step.condition,
                over=start_loop_step.over,
                index=start_loop_step.index,
                loop_block=loop_block,
                loop_step_type=LoopStepType.END,
                loop_type=start_loop_step.loop_type,
                args=start_loop_step.args,
            )
        )
        self._block_stack.pop()
        return self

    def if_(
        self,
        condition: Callable[..., bool] | str,
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Start a new conditional block.

        Steps after this if (and before any else, else_if, or endif) are executed only when the
        `condition` evaluates to `True`. For example:

        # This will run the llm step
        PlanBuilderV2().if_(condition=lambda: True).llm_step(task="Will run").endif()

        # This will not run the llm step
        PlanBuilderV2().if_(condition=lambda: False).llm_step(task="Won't run").endif()

        After opening a conditional block with this if_ method, you must close the block with
        .endif() later in the plan.

        Note: it is if_() rather than if() because if is a keyword in Python.

        Args:
            condition: Condition to evaluate. This can either be a function that returns a boolean
              (as with a traditional if statement) or a string that is evaluated for truth using a
              language model.
            args: Arguments passed to the condition function if it is a function. These are unused
              if the condition is a string.

        """
        parent_block = self._current_conditional_block
        conditional_block = ConditionalBlock(
            clause_step_indexes=[len(self.plan.steps)],
            parent_conditional_block=parent_block,
        )
        self._block_stack.append(conditional_block)
        self.plan.steps.append(
            ConditionalStep(
                condition=condition,
                args=args or {},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=conditional_block,
                clause_index_in_block=0,
                block_clause_type=ConditionalBlockClauseType.NEW_CONDITIONAL_BLOCK,
            )
        )
        return self

    def else_if_(
        self,
        condition: Callable[..., bool],
        args: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add an `else if` clause to the current conditional block.

        Steps after this else_if (and before any 'else' or 'endif') are executed only when the
        `condition` evaluates to `True` and any previous conditions ('if' or 'else if') have been
        evaluated to False. For example:

        # This will run the llm step
        PlanBuilderV2().if_(condition=lambda: False).else_if_(condition=lambda: True)
        .llm_step(task="Will run").endif()

        # This will not run the llm step
        PlanBuilderV2().if_(condition=lambda: False).else_if_(condition=lambda: True)
        .llm_step(task="Won't run").endif()

        # This will not run the llm step
        PlanBuilderV2().if_(condition=lambda: True).else_if_(condition=lambda: True)
        .llm_step(task="Won't run").endif()

        You must ensure that this conditional block is closed with an endif() call later in
        the plan.

        Note: it is else_if_() rather than else_if() to match if_ and else_, which must have an
        underscore because they are keywords in Python.

        Args:
            condition: Condition to evaluate. This can either be a function that returns a boolean
              (as with a traditional if statement) or a string that is evaluated for truth using a
              language model.
            args: Arguments passed to the condition function if it is a function. These are unused
              if the condition is a string.

        """
        if len(self._block_stack) == 0 or not isinstance(self._block_stack[-1], ConditionalBlock):
            raise PlanBuilderError(
                "else_if_ must be called from a conditional block. Please add an if_ first."
            )
        self._block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=condition,
                args=args or {},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=self._block_stack[-1],
                clause_index_in_block=len(self._block_stack[-1].clause_step_indexes) - 1,
                block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
            )
        )
        return self

    def else_(self) -> PlanBuilderV2:
        """Add an `else` clause to the current conditional block.

        Steps after this else (and before any 'endif') are executed only when all previous
        conditions ('if' and any 'else if') have been evaluated to False. For example:

        # This will run the llm step in the else clause
        PlanBuilderV2().if_(condition=lambda: False).llm_step(task="Won't run")
        .else_().llm_step(task="Will run").endif()

        # This will not run the llm step in the else clause
        PlanBuilderV2().if_(condition=lambda: True).llm_step(task="Will run")
        .else_().llm_step(task="Won't run").endif()

        You must ensure that this conditional block is closed with an endif() call later in
        the plan.

        Note: it is else_() rather than else() because else is a keyword in Python.

        """
        if len(self._block_stack) == 0 or not isinstance(self._block_stack[-1], ConditionalBlock):
            raise PlanBuilderError(
                "else_ must be called from a conditional block. Please add an if_ first."
            )
        self._block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=lambda: True,
                args={},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=self._block_stack[-1],
                clause_index_in_block=len(self._block_stack[-1].clause_step_indexes) - 1,
                block_clause_type=ConditionalBlockClauseType.ALTERNATE_CLAUSE,
            )
        )
        return self

    def endif(self) -> PlanBuilderV2:
        """Close the most recently opened conditional block.

        This method must be called to properly close any conditional block that was opened with
        if_(). It marks the end of the conditional logic and allows the plan to continue with
        unconditional steps. For example:

        # Simple if-endif block
        PlanBuilderV2().if_(condition=lambda: False).llm_step(task="Won't run").endif()
        .llm_step(task="Will run")

        # Complex if-else_if-else-endif block
        PlanBuilderV2().if_(condition=lambda: False).llm_step(task="Won't run")
        .else_if_(condition=lambda: False).llm_step(task="Won't run")
        .else_().llm_step(task="Will run")
        .endif()
        .llm_step(task="Always runs after conditional")

        Failing to call endif() after opening a conditional block with if_() will result in an
        error when building the plan.

        """
        if len(self._block_stack) == 0 or not isinstance(self._block_stack[-1], ConditionalBlock):
            raise PlanBuilderError(
                "endif must be called from a conditional block. Please add an if_ first."
            )
        self._block_stack[-1].clause_step_indexes.append(len(self.plan.steps))
        self.plan.steps.append(
            ConditionalStep(
                condition=lambda: True,
                args={},
                step_name=default_step_name(len(self.plan.steps)),
                conditional_block=self._block_stack[-1],
                clause_index_in_block=len(self._block_stack[-1].clause_step_indexes) - 1,
                block_clause_type=ConditionalBlockClauseType.END_CONDITION_BLOCK,
            )
        )
        self._block_stack.pop()
        return self

    def llm_step(
        self,
        *,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
        system_prompt: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that sends a task to an LLM.

        The output from the step is a string (if no output schema is provided) or a structured
        object (if an output schema is provided).

        This just calls a raw LLM without access to tools. If you need to call tools, either use
        a single_tool_agent_step(), a react_agent_step() or an invoke_tool_step().

        Args:
            task: Instruction given to the LLM.
            inputs: Optional additional context for the LLM. Values may reference
                previous step outputs or plan inputs.
            output_schema: Expected schema of the result.
            step_name: Optional explicit name for the step. This allows its output to be referenced
              via StepOutput("name_of_step") rather than by index.
            system_prompt: Optional system prompt for the LLM - allows overriding the default system
              prompt.

        """
        self.plan.steps.append(
            LLMStep(
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
                system_prompt=system_prompt,
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
        """Add a step that invokes a tool directly.

        This is a raw tool call without any LLM involvement. The args passed into this method have
        any references resoled (e.g. Input or StepOutput) and are then directly used to call the
        tool. This should be used when you know the exact arguments you want to pass the tool, and
        you want to avoid the latency / non-determinism of using an LLM.

        Args:
            tool: The id of the tool to invoke, or the Tool instance to invoke.
            args: Arguments passed to the tool. This can include references such as Input and
              StepOutput whose values are resolved at runtime.
            output_schema: Schema of the result. If the tool does not provide a result of this type,
              then a language model will be used to coerce the output into this schema..
            step_name: Optional explicit name for the step. This allows its output to be referenced
              via StepOutput("name_of_step") rather than by index.

        """
        self.plan.steps.append(
            InvokeToolStep(
                tool=tool,
                args=args or {},
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
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
        """Add a step that calls a Python function.

        This step directly calls a python function without any LLM involvement. It is useful for
        incorporating custom logic, calculations, or data transformations into your plan.

        The function is called synchronously if it's a regular function, or awaited if it's
        an async function.

        Args:
            function: The Python function to call. Can be sync or async.
            args: Arguments passed to the function as keyword arguments. This can include
                references such as Input and StepOutput whose values are resolved at runtime.
            output_schema: Schema for the result. If provided and the function output doesn't
                match, a language model will be used to coerce the output into this schema.
            step_name: Optional explicit name for the step. This allows its output to be
                referenced via StepOutput("name_of_step") rather than by index.

        """
        tool_class = tool(function)
        return self.invoke_tool_step(
            tool=tool_class(),
            args=args,
            output_schema=output_schema,
            step_name=step_name,
        )

    def single_tool_agent_step(
        self,
        *,
        tool: str | Tool,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step where an agent uses a single tool to complete a task.

        This creates an LLM agent that has access to exactly one tool and uses it once to complete
        the specified task. The agent can reason about how to use the tool. This is more flexible
        than invoke_tool_step() because the agent can adapt its tool usage based on the task and
        inputs. Use this when you know which tool should be used, but want the agent to determine
        the specific arguments.

        This step outputs the result of the tool call if output schema is not provided. If an output
        schema is provided, then the agent will coerce the result of the tool call to match this
        schema.

        Args:
            tool: The id of the tool or the Tool instance the agent can use to complete the task.
            task: Natural language description of what the agent should accomplish.
            inputs: Optional context data for the agent. This can include references such as
                Input and StepOutput whose values are resolved at runtime and provided as
                context to the agent.
            output_schema: Schema for the result. If provided, the agent will structure its
                output to match this schema.
            step_name: Optional explicit name for the step. This allows its output to be
                referenced via StepOutput("name_of_step") rather than by index.

        """
        self.plan.steps.append(
            SingleToolAgentStep(
                tool=tool,
                task=task,
                inputs=inputs or [],
                output_schema=output_schema,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def react_agent_step(
        self,
        *,
        task: str,
        tools: Sequence[str | Tool] | None = None,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        step_name: str | None = None,
        allow_agent_clarifications: bool = False,
        tool_call_limit: int = 25,
    ) -> PlanBuilderV2:
        """Add a step that uses a ReAct agent with multiple tools.

        The ReAct agent uses reasoning and acting cycles to complete complex tasks
        that may require multiple tool calls and decision making.

        Args:
            task: The task to perform.
            tools: The list of tool IDs or Tool instances to make available to the agent.
            inputs: The inputs to the task. If any of these values are instances of StepOutput or
              Input, the corresponding values will be substituted in when the plan is run.
            output_schema: The schema of the output.
            step_name: Optional name for the step. If not provided, will be auto-generated.
            allow_agent_clarifications: Whether to allow the agent to ask clarifying questions.
            tool_call_limit: Maximum number of tool calls the agent can make.

        """
        self.plan.steps.append(
            ReActAgentStep(
                task=task,
                tools=tools or [],
                inputs=inputs or [],
                output_schema=output_schema,
                allow_agent_clarifications=allow_agent_clarifications,
                tool_call_limit=tool_call_limit,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def user_verify(self, *, message: str, step_name: str | None = None) -> PlanBuilderV2:
        """Add a user confirmation step.

        This pauses plan execution and asks the user to confirm or reject the provided
        message. The plan will only continue if the user confirms. If the user rejects,
        the plan execution will stop with an error. This is useful for getting user approval before
        taking important actions like sending emails, making purchases, or modifying data.

        A UserVerificationClarification is used to get the verification from the user, so ensure you
        have set up handling for this type of clarification in order to use this step. For more
        details, see https://docs.portialabs.ai/understand-clarifications.

        This step outputs True if the user confirms.

        Args:
            message: Text shown to the user for confirmation. This can include references
                such as Input and StepOutput whose values are resolved at runtime before
                being shown to the user.
            step_name: Optional explicit name for the step. This allows its output to be
                referenced via StepOutput("name_of_step") rather than by index.

        """
        self.plan.steps.append(
            UserVerifyStep(
                message=message,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def user_input(
        self,
        *,
        message: str,
        options: list[Serializable] | None = None,
        step_name: str | None = None,
    ) -> PlanBuilderV2:
        """Add a step that requests input from the user.

        This pauses plan execution and prompts the user to provide input. If options are
        provided, the user must choose from the given choices (multiple choice). If no
        options are provided, the user can enter free-form text.

        A Clarification (either InputClarification or MultipleChoiceClarification) is used to get
        the input from the user, so ensure you have set up handling for the required type of
        clarification in order to use this step. For more details, see
        https://docs.portialabs.ai/understand-clarifications.

        The user's response becomes the output of this step and can be referenced by
        subsequent steps in the plan.

        Args:
            message: Instruction or question shown to the user. This can include references
                such as Input and StepOutput whose values are resolved at runtime before
                being shown to the user.
            options: List of choices for multiple choice prompts. If None, the user can
                provide free-form text input. If provided, the user must select from
                these options.
            step_name: Optional explicit name for the step. This allows its output to be
                referenced via StepOutput("name_of_step") rather than by index.

        """
        self.plan.steps.append(
            UserInputStep(
                message=message,
                options=options,
                step_name=step_name or default_step_name(len(self.plan.steps)),
                conditional_block=self._current_conditional_block,
            )
        )
        return self

    def add_step(self, step: StepV2) -> PlanBuilderV2:
        """Add a pre-built step to the plan.

        This allows you to integrate custom step types that you've created by subclassing
        StepV2, or to reuse steps that were created elsewhere. The step is added as-is
        to the plan without any modification.

        Args:
            step: A pre-built step instance that inherits from StepV2.

        """
        self.plan.steps.append(step)
        return self

    def add_steps(
        self,
        plan: PlanV2 | Iterable[StepV2],
        input_values: dict[str, Any] | None = None,
    ) -> PlanBuilderV2:
        """Add multiple steps or merge another plan into this builder.

        This allows you to compose plans by merging smaller plans together, or to add
        a sequence of pre-built steps all at once. When merging a PlanV2, both the
        steps and the plan inputs are merged into the current builder.

        This is useful for creating reusable sub-plans that can be incorporated into
        larger workflows.

        Args:
            plan: Either a complete PlanV2 to merge (including its steps and inputs),
                or any iterable of StepV2 instances to add to the current plan.
            input_values: Optional mapping of inputs in the sub-plan to values. This is
                only used when plan is a PlanV2, and is useful if a sub-plan has an input
                and you want to provide a value for it from a step in the top-level plan.
                For example:

            ```python
                sub_plan = builder.input(name="input_name").build()
                top_plan = builder.llm_step(step_name="llm_step", task="Task")
                           .add_steps(sub_plan, input_values={"input_name": StepOutput("llm_step")})
                           .build()
            ```

            input_values: Optional mapping of input names to default values. Only used
                when plan is a PlanV2. These values will be set as default values for
                the corresponding plan inputs.

        Raises:
            PlanBuilderError: If duplicate input names are detected when merging plans,
                or if you try to provide values for inputs that don't exist in the
                sub-plan.

        """
        if isinstance(plan, PlanV2):
            # Ensure there are no duplicate plan inputs
            existing_input_names = {p.name for p in self.plan.plan_inputs}
            for _input in plan.plan_inputs:
                if _input.name in existing_input_names:
                    raise PlanBuilderError(f"Duplicate input {_input.name} found in plan.")
            self.plan.plan_inputs.extend(plan.plan_inputs)
            self.plan.steps.extend(plan.steps)
        else:
            self.plan.steps.extend(plan)

        if input_values and isinstance(plan, PlanV2):
            allowed_input_names = {p.name for p in plan.plan_inputs}
            for input_name, input_value in input_values.items():
                if input_name not in allowed_input_names:
                    raise PlanBuilderError(
                        f"Tried to provide value for input {input_name} not found in "
                        "sub-plan passed into add_steps()."
                    )
                for plan_input in self.plan.plan_inputs:
                    if plan_input.name == input_name:
                        plan_input.value = input_value
                        break

        return self

    def final_output(
        self,
        output_schema: type[BaseModel] | None = None,
        summarize: bool = False,
    ) -> PlanBuilderV2:
        """Define the final output schema for the plan.

        This configures how the plan's final result should be structured. The final output is
        automatically derived from the last step's output, but if output_schema is provided and
        the last step's output does not already match this schema, an LLM will be used to coerce
        the output into this schema. If summarize is True, a summary of the plan run will also be
        included as part of the final output.

        Args:
            output_schema: Pydantic model class that defines the structure of the final
                output.
            summarize: Whether to also generate a human-readable summary of the final
                output in addition to the structured result.

        """
        self.plan.final_output_schema = output_schema
        self.plan.summarize = summarize
        return self

    def build(self) -> PlanV2:
        """Finalize and return the built plan.

        Raises:
            PlanBuilderError: If there are any issues when building the plan (e.g. an if block is
                opened without being closed).

        """
        if len(self._block_stack) > 0:
            raise PlanBuilderError(
                "All blocks must be closed. Please add an endif or endloop for all blocks."
            )

        step_type_counts: dict[str, int] = {}
        for step in self.plan.steps:
            step_type = step.__class__.__name__
            step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1

        telemetry = ProductTelemetry()
        telemetry.capture(
            PlanV2BuildTelemetryEvent(
                plan_length=len(self.plan.steps), step_type_counts=step_type_counts
            )
        )

        return self.plan
