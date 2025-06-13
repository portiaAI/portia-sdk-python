"""Tool decorator for creating tools from functions."""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any, TypeVar, get_origin, get_args, get_type_hints

from pydantic import BaseModel, Field, create_model

from portia.tool import Tool, ToolRunContext

# Import Annotated from appropriate module based on Python version
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
else:
    from collections.abc import Callable

# Type variables for the decorator
P = inspect.Parameter
T = TypeVar("T")


def tool(fn: Callable[..., T] | Callable[..., Awaitable[T]]) -> type[Tool[T]]:
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
    description = (fn.__doc__ or "").strip()

    # Generate tool properties
    tool_id = func_name
    tool_name = _snake_to_title_case(func_name)

    # Get function signature and type hints
    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn)

    # Create args schema from function parameters
    args_schema = _create_args_schema(sig, type_hints, func_name, fn)

    # Determine output schema from return type
    output_schema = _create_output_schema(type_hints, func_name)

    # Create the Tool class dynamically
    def make_run_method() -> Callable:
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

    class FunctionTool(Tool[T]):
        """Dynamically created tool from function."""

        def __init__(self, **data: Any) -> None:
            # Set all the class attributes from the closure
            super().__init__(
                id=tool_id,
                name=tool_name,
                description=description,
                args_schema=args_schema,
                output_schema=output_schema,
                **data,
            )

        run = make_run_method()

    # Set class name for better debugging
    FunctionTool.__name__ = f"{_snake_to_title_case(func_name).replace(' ', '')}Tool"
    FunctionTool.__qualname__ = FunctionTool.__name__

    return FunctionTool


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


def _create_args_schema(
    sig: inspect.Signature, type_hints: dict[str, Any], func_name: str, func: Callable
) -> type[BaseModel]:
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
    param_annotation: Any,
    param: inspect.Parameter,
    param_name: str,
    func_name: str,
) -> tuple[Any, Field]:
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
        args = get_args(param_annotation)
        if not args:
            # Malformed Annotated, treat as Any
            param_type = Any
            description = _extract_param_description(None, param_name, func_name)
            field_info = Field(default=default, description=description)
        else:
            # First arg is the actual type
            param_type = args[0]

            # Look for description in metadata
            description = None
            field_kwargs = {}

            for metadata in args[1:]:
                if isinstance(metadata, str):
                    # Annotated[str, "description"]
                    description = metadata
                elif hasattr(metadata, "description") and hasattr(metadata, "default"):
                    # Annotated[str, Field(description="...")]
                    description = metadata.description
                    # Copy specific field validation properties we care about
                    validation_attrs = [
                        "gt",
                        "ge",
                        "lt",
                        "le",
                        "min_length",
                        "max_length",
                        "regex",
                        "allow_inf_nan",
                    ]
                    for attr in validation_attrs:
                        if hasattr(metadata, attr):
                            value = getattr(metadata, attr)
                            if value is not None:
                                field_kwargs[attr] = value

                    # Also check for constraints in metadata (for pydantic v2)
                    if hasattr(metadata, "metadata"):
                        for constraint in metadata.metadata:
                            if hasattr(constraint, "min_length"):
                                field_kwargs["min_length"] = constraint.min_length
                            elif hasattr(constraint, "max_length"):
                                field_kwargs["max_length"] = constraint.max_length
                            elif hasattr(constraint, "gt"):
                                field_kwargs["gt"] = constraint.gt
                            elif hasattr(constraint, "ge"):
                                field_kwargs["ge"] = constraint.ge
                            elif hasattr(constraint, "lt"):
                                field_kwargs["lt"] = constraint.lt
                            elif hasattr(constraint, "le"):
                                field_kwargs["le"] = constraint.le

                    # Use Field's default if specified and param has no default
                    if (
                        metadata.default is not ...
                        and param.default == inspect.Parameter.empty
                    ):
                        default = metadata.default

            # Use extracted description or fallback
            if description is None:
                description = _extract_param_description(None, param_name, func_name)

            field_info = Field(default=default, description=description, **field_kwargs)

        return param_type, field_info

    # Regular type annotation without special metadata
    param_type = param_annotation
    description = _extract_param_description(None, param_name, func_name)
    field_info = Field(default=default, description=description)

    return param_type, field_info


def _create_output_schema(
    type_hints: dict[str, Any], func_name: str
) -> tuple[str, str]:
    """Create output schema tuple from return type annotation."""
    return_type = type_hints.get("return", Any)

    # Convert type to string representation
    type_str = (
        return_type.__name__ if hasattr(return_type, "__name__") else str(return_type)
    )

    # Create description
    description = f"Output from {func_name} function"

    return (type_str, description)


def _extract_param_description(
    sig: inspect.Signature | None,  # noqa: ARG001
    param_name: str,
    func_name: str,
) -> str:
    """Extract parameter description from function docstring.

    This is a simple implementation that looks for basic patterns.
    Could be enhanced to parse more sophisticated docstring formats.
    """
    # For now, return a basic description
    # In a real implementation, you might parse docstrings more thoroughly
    return f"Parameter {param_name} for {func_name}"
