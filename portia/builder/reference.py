"""References to values in a plan.

This module provides reference classes that allow you to reference variables in your plan whose
values are not known when the plan is built, such as outputs from previous steps or user-provided
inputs. When the plan is run, these references are then resolved to their actual values.

The Reference class provides an interface for all reference types. It then has the following
implementations:
    StepOutput: References the output value from a previous step.
    Input: References a user-provided input value.

Example:
    ```python
    from portia.builder import PlanBuilderV2
    from portia.builder.reference import StepOutput, Input

    builder = PlanBuilderV2()
    builder.input("user_query", description="The user's search query")

    # Step 1: Search for information
    builder.step("search", tools=[search_tool], inputs={"query": Input("user_query")})

    # Step 2: Process the search results
    builder.step("process", tools=[llm_tool], inputs={"data": StepOutput("search")})
    ```

"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import signature
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover


import pydash
from pydantic import BaseModel, ConfigDict, Field

from portia.logger import logger

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.run_context import RunContext, StepOutputValue

_KWARGS_ARG_REGEX = re.compile(r"^\s*(\w+)\s*=\s*(.+)\s*$")
_POSITIONAL_ARG_REGEX = re.compile(r"^\s*(.+)\s*$")
_PARENTHESIS_STRING_REGEX = re.compile(r"(?:{{ )?(\w+)\((.*)\)(?: }})?")


def default_step_name(step_index: int) -> str:
    """Generate a default name for a step based on its index.

    Args:
        step_index: The zero-based index of the step in the plan.

    Returns:
        A string in the format `"step_{index}" (e.g., "step_0", "step_1")`.

    """
    return f"step_{step_index}"


T = TypeVar("T", bound="Reference")


def string_to_none(input_str: str) -> None:
    """Convert a string to None."""
    if input_str.strip() == "None":
        return
    raise ValueError(f"Invalid none string: {input_str}")


def string_to_bool(input_str: str) -> bool:
    """Convert a string to a boolean."""
    if input_str.strip() == "True":
        return True
    if input_str.strip() == "False":
        return False
    raise ValueError(f"Invalid boolean string: {input_str}")


def parenthesis_string_to_str(input_str: str) -> str:
    """Convert a parenthesis string to a string."""
    if (input_str.startswith('"') and input_str.endswith('"')) or (
        input_str.startswith("'") and input_str.endswith("'")
    ):
        return input_str[1:-1]
    raise ValueError(f"Invalid parenthesis string: {input_str}")


_DEFAULT_CONVERTERS: list[Callable[[str], Any]] = [
    parenthesis_string_to_str,
    int,
    float,
    string_to_none,
    string_to_bool,
]


class Reference(BaseModel, ABC):
    """Abstract base class for all reference types in Portia plans.

    References allow you to dynamically reference values that will be resolved at runtime,
    such as outputs from previous steps or user-provided inputs. This enables building
    flexible workflows where later steps can depend on the results of earlier operations.

    This is an abstract base class that defines the interface all reference types must
    implement. Concrete implementations include StepOutput and Input.

    """

    # Allow setting temporary/mock attributes in tests (e.g. patch.object(..., "get_value"))
    # Without this, Pydantic v2 prevents setting non-field attributes on instances.
    model_config = ConfigDict(extra="allow")
    # Converters are used to convert strings to the appropriate type after string parsing,
    # in order of precedence. First to convert without raising valueerror is the type that is used
    # when rebuilding the object. If you need to support a new type or custom type, add a converter
    _converters: ClassVar[list[Callable[[str], Any]]] = _DEFAULT_CONVERTERS

    @abstractmethod
    def get_legacy_name(self, plan: PlanV2) -> str:
        """Get the reference name for compatibility with legacy Portia plans.

        Args:
            plan: The PlanV2 instance containing the reference.

        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_value(self, run_data: RunContext) -> Any | None:  # noqa: ANN401
        """Resolve and return the value this reference points to.

        Args:
            run_data: The runtime context containing step outputs, inputs, and other
                execution data needed to resolve the reference.

        Returns:
            The resolved value, or None if the reference cannot be resolved.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_str(cls: type[T], input_str: str) -> T:
        """Create a reference from a string representation.

        Args:
            input_str: The string representation of the reference.

        Returns:
            The reference object.

        Examples:
            ```python
            StepOutput.from_str("StepOutput(step_name, path='field.name')")
            StepOutput.from_str("{{ StepOutput(0, path='field.name') }}")
            Input.from_str("Input(input_name, path='field.name')")
            ```

        """
        parsed_args, kwargs = cls._parse_argument(input_str)
        init_kwargs = list(signature(cls.__init__).parameters.values())[1:]  # exclude self
        for param, init_kwarg in zip(parsed_args, init_kwargs, strict=False):
            kwargs[init_kwarg.name] = param
        kwargs = {k: cls._convert_argument(v) for k, v in kwargs.items()}
        return cls(**kwargs)

    @classmethod
    def _parse_argument(cls, input_str: str) -> tuple[list[str], dict[str, Any]]:
        """Parse the arguments from the input string into positional and keyword arguments."""
        # Use regex to capture class name and arguments, optionally handling {{ }} wrapper
        match = re.search(_PARENTHESIS_STRING_REGEX, input_str)
        if not match:
            raise ValueError(f"Invalid input string format: {input_str}")
        class_name = match.group(1)
        if class_name != cls.__name__:
            raise ValueError(f"Invalid input string format: {input_str}")
        raw_args = match.group(2).strip()
        if raw_args is None or raw_args == "":
            return [], {}
        args = raw_args.split(",")
        must_be_closed = ['"', "'"]
        must_be_matched = ["{", "}", "[", "]", "(", ")"]
        counts = {char: input_str.count(char) for char in must_be_closed + must_be_matched}
        if (
            counts["{"] != counts["}"]
            or counts["["] != counts["]"]
            or counts["("] != counts[")"]
            or counts["'"] % 2 != 0
            or counts['"'] % 2 != 0
        ):
            raise ValueError(f"Invalid input string format: {input_str}")
        kwargs = {}
        parsed_args = []
        for arg in args:
            if arg.strip() == "":
                raise ValueError(f"Invalid input string format: {input_str}")
            if "=" in arg:
                matcher = _KWARGS_ARG_REGEX.match(arg)
                if not matcher:
                    raise ValueError(f"Invalid input string format: {input_str}")
                key, value = matcher.groups()
                kwargs[key.strip()] = value.strip()
            elif matcher := _POSITIONAL_ARG_REGEX.match(arg):
                parsed_args.append(matcher.group(1).strip())
            else:
                # this shouldn't be reachable
                raise ValueError(f"Invalid input string format: {input_str}")  # pragma: no cover
        return parsed_args, kwargs

    @classmethod
    def _convert_argument(cls, input_str: str) -> Any:  # noqa: ANN401
        """Parse an argument from a string."""
        for converter in cls._converters:
            try:
                return converter(input_str)
            except ValueError:
                continue
        return input_str


class StepOutput(Reference):
    """A reference to the output of a previous step in the plan.

    StepOutput allows you to reference the output value produced by an earlier step,
    enabling you to create workflows where later steps depend on the results of
    previous operations. The reference is resolved at runtime during plan execution.

    You can reference a step either by its name (string) or by its position (integer index).
    Step indices are zero-based, so the first step is index 0. Negative indices are
    also supported, with -1 referring to the last step in the plan, -2 the second to
    last, and so on.

    Example:
        ```python
        from portia.builder import PlanBuilderV2
        from portia.builder.reference import StepOutput

        builder = PlanBuilderV2()

        # Create a step that produces some output using a search tool
        builder.single_tool_agent_step("search", "search_tool",
                                      inputs=["Find Python tutorials"])

        # Reference the output by name
        builder.llm_step("analyze", task="Analyze the search results",
                        inputs=[StepOutput("search")])

        # Reference the output by index (0 = first step)
        builder.llm_step("summarize", task="Summarize the data",
                        inputs=[StepOutput(0)])

        # Access nested fields using path notation (i.e. take the .title attribute from the output
        # of the search step)
        builder.llm_step("process", task="Extract title from results",
                        inputs=[StepOutput("search", path="title")])

        # Access deeply nested fields
        builder.llm_step("extract", task="Get street address",
                        inputs=[StepOutput("user_data", path="address.street")])
        ```

    """

    step: str | int = Field(
        description="The step to reference the output of. If a string is provided, this will be "
        "used to find the step by name. If an integer is provided, this will be used to find the "
        "step by index (steps are 0-indexed)."
    )
    path: str | None = Field(
        default=None,
        description=(
            "Optional path to extract a specific field or substructure from the step output. "
            "Uses dot notation for all access types: "
            "1. Object attributes: 'field.subfield' to access object properties, "
            "2. Array indexes: 'items.0.name' to access list/array elements by index, "
            "3. Dict keys: 'data.key' to access dictionary values. "
            "These can be combined: 'results.0.user.address.street'."
        ),
    )
    full: bool = Field(
        default=False,
        description=(
            "Whether to return the full step output values as a list. "
            "Used in the case of loop steps."
        ),
    )

    def __init__(self, step: str | int, path: str | None = None, full: bool = False) -> None:
        """Initialize a reference to a step's output."""
        super().__init__(step=step, path=path, full=full)  # type: ignore[call-arg]

    @override
    def get_legacy_name(self, plan: PlanV2) -> str:
        """Get the reference name for compatibility with legacy Portia plans."""
        return plan.step_output_name(self.step)

    def __str__(self) -> str:
        """Get the string representation of the step output."""
        # We use double braces around the StepOutput to allow string interpolation for StepOutputs
        # used in PlanBuilderV2 steps.
        # The double braces are used when the plan is running to template the StepOutput value so it
        # can be substituted at runtime.
        step_repr = f"'{self.step}'" if isinstance(self.step, str) else str(self.step)
        base_repr = f"StepOutput({step_repr}"
        if self.path:
            base_repr += f", path='{self.path}'"
        if self.full:
            base_repr += ", full=True"
        return f"{{{{ {base_repr}) }}}}"

    @override
    def get_value(self, run_data: RunContext) -> Any | None:
        """Resolve and return the output value from the referenced step.

        If a path is provided, it will be used to navigate the step output
        to extract the desired substructure.

        Note:
            If the step cannot be found, a warning is logged and None is returned.
            The step is matched by either name (string) or index (integer).
            If the path navigation fails, the exception is propagated.

        """
        # Get the base step output value
        to_parse = None
        total_steps = len(run_data.plan.steps)
        if self.full:
            to_parse = [
                step_output.value
                for step_output in run_data.step_output_values
                if self._match_output(step_output, total_steps)
            ]
        else:
            for step_output in run_data.step_output_values[::-1]:
                if self._match_output(step_output, total_steps):
                    to_parse = [step_output.value]
                    break
        if to_parse is None:
            logger().warning(f"Output value for step {self.step} not found")
            return None
        if self.path:
            parsed = [pydash.get(step_output, self.path) for step_output in to_parse]
        else:
            parsed = to_parse
        return parsed if self.full else parsed[0]

    def _match_output(self, step_output: StepOutputValue, total_steps: int) -> bool:
        """Match the step output to the step stored in the reference.

        Args:
            step_output: The step output to compare against.
            total_steps: Total number of step outputs available, used for resolving
                negative indices.

        """
        if isinstance(self.step, int):
            index = self.step if self.step >= 0 else total_steps + self.step
            return step_output.step_num == index
        return step_output.step_name == self.step

    def get_description(self, run_data: RunContext) -> str:
        """Get the description of the step output."""
        total_steps = len(run_data.plan.steps)
        for step_output in run_data.step_output_values:
            if self._match_output(step_output, total_steps):
                return step_output.description
        return ""


class Input(Reference):
    """A reference to a user-provided plan input.

    Input allows you to reference values that are provided at runtime when the plan
    is executed, rather than when the plan is built. This enables creating reusable
    plans that can work with different input values each time they run.

    Plan inputs are defined using the PlanBuilder.input() method when building your
    plan, and their values are provided when calling portia.run() or portia.arun().
    The Input reference is then resolved to the actual value during execution.

    Example:
        ```python
        from portia.builder import PlanBuilderV2
        from portia.builder.reference import Input

        builder = PlanBuilderV2()

        # Define an input that will be provided at runtime
        builder.input("user_query", description="The user's search query")
        builder.input("max_results", description="Maximum number of results")

        # Use the inputs in steps
        builder.step("search",
                    tools=[search_tool],
                    inputs={
                        "query": Input("user_query"),
                        "limit": Input("max_results")
                    })

        # When running the plan, provide the input values:
        # portia.run(plan, inputs={"user_query": "Python tutorials", "max_results": 10})
        ```

    """

    name: str = Field(description="The name of the input as defined in the plan.")
    path: str | None = Field(
        default=None,
        description=(
            "Optional path to extract a specific field or substructure from the input value. "
            "Uses dot notation for all access types: "
            "1. Object attributes: 'field.subfield' to access object properties, "
            "2. Array indexes: 'items.0.name' to access list/array elements by index, "
            "3. Dict keys: 'data.key' to access dictionary values. "
            "These can be combined: 'results.0.user.address.street'."
        ),
    )

    def __init__(self, name: str, path: str | None = None) -> None:
        """Initialize a reference to a plan input."""
        # We have this __init__ method so users can create an Input reference without having to
        # specify name= in the constructor.
        super().__init__(name=name, path=path)  # type: ignore[call-arg]

    @override
    def get_legacy_name(self, plan: PlanV2) -> str:
        """Get the reference name for compatibility with legacy Portia plans."""
        return self.name

    @override
    def get_value(self, run_data: RunContext) -> Any | None:
        """Resolve and return the user-provided input value.

        Returns:
            The user-provided value for this input, or None if not found.

        Note:
            If the input value is itself a Reference, it will be recursively resolved.

        """
        plan_input = next(
            (_input for _input in run_data.plan.plan_inputs if _input.name == self.name), None
        )
        if not plan_input:
            logger().warning(f"Input {self.name} not found in plan")
            return None
        local_data_value = run_data.plan_run.plan_run_inputs.get(self.name)
        if not local_data_value:
            logger().warning(f"Value not found for input {self.name}")
            return None
        value = local_data_value.get_value()
        resolved_value = (
            value.get_value(run_data=run_data) if isinstance(value, Reference) else value
        )

        # If there is a path, use pydash to traverse the object
        return pydash.get(resolved_value, self.path) if self.path else resolved_value

    def __str__(self) -> str:
        """Get the string representation of the input."""
        # We use double braces around the Input to allow string interpolation for Inputs used
        # in PlanBuilderV2 steps. The double braces are used when the plan is running to template
        # the input value so it can be substituted at runtime.
        if self.path:
            return f"{{{{ Input('{self.name}', path='{self.path}') }}}}"
        return f"{{{{ Input('{self.name}') }}}}"
