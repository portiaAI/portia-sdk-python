from inspect import getdoc, getfullargspec
from typing import Any, Callable, TypeVar, get_type_hints

from pydantic import create_model

from portia.tool import Tool, ToolRunContext

T = TypeVar("T")


def tool(func: Callable[..., Any]) -> Tool:
    """Decorator that converts a Python function into a Portia Tool object.

    The decorator extracts:
    - Function arguments and type annotations to create an input arg schema
    - Return type annotation for output schema
    - Docstring for Tool description
    - Function name for Tool id and name

    Args:
        func: The function to convert into a Tool

    Returns:
        Tool: A Portia Tool object configured based on the function

    """
    # Get function metadata
    func_name = func.__name__
    docstring = getdoc(func) or ""
    type_hints = get_type_hints(func)
    argspec = getfullargspec(func)

    # Create input schema from function arguments
    schema_fields = {}
    for arg in argspec.args:
        if arg != "self":  # Skip self parameter if present
            field_type = type_hints.get(arg, Any)
            schema_fields[arg] = (field_type, ...)  # ... means required field

    # Create Pydantic model for input schema
    input_schema = create_model(
        f"{func_name}InputSchema",
        **schema_fields,
    )

    # Get return type for output schema
    return_type = type_hints.get("return", Any)
    return_type_str = return_type.__name__ if hasattr(return_type, "__name__") else str(return_type)

    # Create a concrete Tool class that implements run
    class FunctionTool(Tool):
        def run(self, ctx: ToolRunContext, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    # Create and return the Tool instance
    return FunctionTool(
        id=func_name,
        name=func_name,
        description=docstring,
        args_schema=input_schema,
        output_schema=(return_type_str, docstring),
        should_summarize=False,
    )


def tool_with_summary(func: Callable[..., Any]) -> Tool:
    """Decorator that converts a Python function into a Portia Tool object with summarization enabled.

    This is identical to the @tool decorator but sets should_summarize=True.

    Args:
        func: The function to convert into a Tool

    Returns:
        Tool: A Portia Tool object configured based on the function

    """
    tool_obj = tool(func)
    tool_obj.should_summarize = True
    return tool_obj
