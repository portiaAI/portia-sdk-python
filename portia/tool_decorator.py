"""Tool decorator for creating tools from functions."""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from collections.abc import Callable

from portia.common import SERIALIZABLE_TYPE_VAR
from portia.tool import Tool, ToolRunContext

# Type variables for the decorator
P = inspect.Parameter
T = SERIALIZABLE_TYPE_VAR


class DecoratedTool(Tool[T]):
    """Decorated tool class."""

    id: str = Field(default="DecoratedTool", description="ID of the tool")
    name: str = Field(default="DecoratedTool", description="Name of the tool")
    description: str = Field(default="DecoratedTool", description="Description of the tool")
    args_schema: type[BaseModel] = Field(default=BaseModel, description="Input schema of the tool")
    output_schema: tuple[str, str] = Field(
        default=("str", "Output from the tool"), description="Output schema of the tool"
    )


# Create the Tool class dynamically
def make_run_method(sig: inspect.Signature, fn: Callable) -> Callable:
    """Make the run method for the tool."""

    def run(self: Tool[T], ctx: ToolRunContext, **kwargs: Any) -> T:  # noqa: ARG001
        """Execute the original function."""
        # Filter out 'ctx' parameter if the function doesn't expect it
        func_params = set(sig.parameters.keys())
        if "ctx" in func_params:
            kwargs["ctx"] = ctx
        elif "context" in func_params:
            kwargs["context"] = ctx

        # Call the original function with filtered kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
        return fn(**filtered_kwargs)

    return run


def make_arun_method(sig: inspect.Signature, fn: Callable) -> Callable:
    """Make the arun method for the tool."""

    async def arun(self: Tool[T], ctx: ToolRunContext, **kwargs: Any) -> T:  # noqa: ARG001
        """Execute the original function."""
        # Filter out 'ctx' parameter if the function doesn't expect it
        func_params = set(sig.parameters.keys())
        if "ctx" in func_params:
            kwargs["ctx"] = ctx
        elif "context" in func_params:
            kwargs["context"] = ctx

        # Call the original function with filtered kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
        if inspect.iscoroutinefunction(fn):
            return await fn(**filtered_kwargs)
        # this line actually shouldn't be reached but the type checker doesn't know that
        return fn(**filtered_kwargs)  # pragma: no cover

    return arun


def make_not_implemented_method() -> Callable:
    """Make a run method that raises a NotImplementedError."""

    def not_implemented_function(self: Tool[T], ctx: ToolRunContext, **kwargs: Any) -> T:
        """Not implemented. We should not be calling this method."""
        raise NotImplementedError("Async Tool should use the arun method.")

    return not_implemented_function


def tool(fn: Callable[..., T]) -> type[DecoratedTool]:
    """Convert a function into a Tool class.

    This decorator automatically creates a Tool subclass from a function by:
    - Using the function's docstring as the tool description
    - Creating an ID and name based on the function name
    - Generating input schema from function parameters and type hints
    - Determining output schema from return type annotation

    Example:
        @tool
        def add_numbers(a: int, b: int) -> int:
            \"\"\"Add two numbers together.\"\"\"
            return a + b

    Args:
        fn: The function to convert to a Tool class

    Returns:
        A Tool subclass that wraps the original function

    Raises:
        ValueError: If the function has invalid signature or return type

    """
    # Validate function
    _validate_function(fn)

    # Extract function metadata
    func_name = fn.__name__
    tool_description = (fn.__doc__ or "").strip()

    # Generate tool properties
    tool_id = func_name
    tool_name = _snake_to_title_case(func_name)

    # Get function signature and type hints
    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn)

    # Create args schema from function parameters
    tool_args_schema = _create_args_schema(sig, func_name, fn)

    # Determine output schema from return type
    tool_output_schema = _create_output_schema(type_hints, func_name)

    class FunctionTool(DecoratedTool):
        """Dynamically created tool from function."""

        id: str = tool_id
        name: str = tool_name
        description: str = tool_description
        args_schema: type[BaseModel] = tool_args_schema
        output_schema: tuple[str, str] = tool_output_schema

        run = make_run_method(sig, fn)

    class AsyncFunctionTool(DecoratedTool):
        """Dynamically created async tool from function."""

        id: str = tool_id
        name: str = tool_name
        description: str = tool_description
        args_schema: type[BaseModel] = tool_args_schema
        output_schema: tuple[str, str] = tool_output_schema

        run = make_not_implemented_method()
        arun = make_arun_method(sig, fn)

    class_to_return = AsyncFunctionTool if inspect.iscoroutinefunction(fn) else FunctionTool
    # Set class name for better debugging
    class_to_return.__name__ = f"{_snake_to_title_case(func_name).replace(' ', '')}Tool"
    class_to_return.__qualname__ = class_to_return.__name__

    return class_to_return


def _validate_function(fn: Callable) -> None:
    """Validate that the function is suitable for conversion to a tool."""
    if not callable(fn):
        raise TypeError("Decorated object must be callable")

    # Check that function has type hints for return type
    type_hints = get_type_hints(fn)
    if "return" not in type_hints:
        raise ValueError(f"Function '{fn.__name__}' must have a return type annotation")


def _snake_to_title_case(snake_str: str) -> str:
    """Convert snake_case to Title Case."""
    return " ".join(word.capitalize() for word in snake_str.split("_"))


def _create_args_schema(sig: inspect.Signature, func_name: str, func: Callable) -> type[BaseModel]:
    """Create a Pydantic schema from function parameters."""
    fields = {}

    # Try to get type hints with extras to preserve Annotated metadata
    try:
        type_hints_with_extras = get_type_hints(func, include_extras=True)
    except (TypeError, AttributeError, NameError):
        type_hints_with_extras = {}

    for param_name, param in sig.parameters.items():
        # Skip context parameters
        if param_name in ("ctx", "context"):
            continue

        # First try to get annotation with extras (for Annotated support)
        param_annotation = type_hints_with_extras.get(param_name)
        if param_annotation == ToolRunContext and param_name not in ("ctx", "context"):
            raise ValueError(
                f"Tool {func_name} has a ToolRunContext parameter that is not named 'ctx' "
                f"or 'context'. This is not allowed as it causes errors when the tool "
                "inputs are validated."
            )

        # Fall back to raw annotation from signature
        if param_annotation is None:
            param_annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

        # Extract type and field info from annotation
        param_type, field_info = _extract_type_and_field_info(
            param_annotation, param, param_name, func_name
        )

        fields[param_name] = (param_type, field_info)

    # Create the schema class
    schema_class_name = f"{_snake_to_title_case(func_name).replace(' ', '')}Schema"
    return create_model(
        schema_class_name,
        **fields,
        __base__=BaseModel,
    )


def _extract_type_and_field_info(
    param_annotation: type | object,
    param: inspect.Parameter,
    param_name: str,
    func_name: str,
) -> tuple[type | object, FieldInfo]:
    """Extract type and field information from parameter annotation.

    Supports:
    - Annotated with string: name: Annotated[str, "description"]
    - Annotated with Field: name: Annotated[str, Field(description="...")]
    - Regular type hints: name: str
    """
    # Determine default value
    default = ... if param.default == inspect.Parameter.empty else param.default

    # Check if annotation is Annotated type
    origin = get_origin(param_annotation)
    if origin is Annotated:
        return _process_annotated_type(param_annotation, param_name, func_name, default)

    # Regular type annotation without special metadata
    return _process_regular_type(param_annotation, param_name, func_name, default)


def _process_annotated_type(
    param_annotation: type | object,
    param_name: str,
    func_name: str,
    default: object,
) -> tuple[type | object, FieldInfo]:
    """Process Annotated type annotations."""
    args = get_args(param_annotation)
    if not args:
        # Malformed Annotated, treat as Any
        param_type = Any
        description = _form_param_description(param_name, func_name)
        field_info = cast("FieldInfo", Field(default=default, description=description))
        return param_type, field_info

    # First arg is the actual type
    param_type = args[0]

    # Extract metadata from annotations, it is an annotation therefore there is a second argument
    annotation_metadata = args[1]

    field_info: FieldInfo
    match annotation_metadata:
        case str():
            field_info = cast("FieldInfo", Field(default=default, description=annotation_metadata))
        case FieldInfo():
            field_info = cast("FieldInfo", annotation_metadata)
            field_info.default = default
        case _:
            raise ValueError(f"Unsupported annotation metadata: {annotation_metadata}")
    return param_type, field_info


def _process_regular_type(
    param_annotation: type | object,
    param_name: str,
    func_name: str,
    default: object,
) -> tuple[type | object, FieldInfo]:
    """Process regular type annotations without metadata."""
    param_type = param_annotation
    description = _form_param_description(param_name, func_name)
    field_info = cast("FieldInfo", Field(default=default, description=description))
    return param_type, field_info


def _create_output_schema(type_hints: dict[str, type | object], func_name: str) -> tuple[str, str]:
    """Create output schema tuple from return type annotation."""
    return_type = type_hints.get("return", Any)

    # Convert type to string representation
    type_str = return_type.__name__ if hasattr(return_type, "__name__") else str(return_type)  # type: ignore[attr-defined]

    # Create description
    description = f"Output from {func_name} function"

    return (type_str, description)


def _form_param_description(param_name: str, func_name: str) -> str:
    """Format parameter description using the func_name.

    This is used for arg that are not annotated with string or FieldInfo.
    """
    return f"Parameter {param_name} for {func_name}"
