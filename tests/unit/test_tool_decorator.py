"""Tests for the tool decorator."""

import os
import unittest.mock
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from portia.errors import InvalidToolDescriptionError, ToolHardError, ToolSoftError
from portia.tool import ToolRunContext
from portia.tool_decorator import (
    _create_args_schema,
    _extract_type_and_field_info,
    tool,
)
from tests.utils import get_test_tool_context


def test_basic_tool_decorator() -> None:
    """Test basic usage of the tool decorator."""

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    # Create an instance of the tool
    tool_instance = add_numbers()

    # Check basic properties
    assert tool_instance.id == "add_numbers"
    assert tool_instance.name == "Add Numbers"
    assert tool_instance.description == "Add two numbers together."
    assert tool_instance.output_schema == ("int", "Output from add_numbers function")

    # Test args schema
    schema = tool_instance.args_schema
    assert issubclass(schema, BaseModel)
    assert hasattr(schema, "model_fields")
    assert "a" in schema.model_fields
    assert "b" in schema.model_fields

    # Test running the tool
    ctx = get_test_tool_context()
    result = tool_instance.run(ctx, a=3, b=4)
    assert result == 7


def test_tool_with_optional_parameters() -> None:
    """Test tool decorator with optional parameters."""

    @tool
    def greet_user(name: str, greeting: str = "Hello") -> str:
        """Greet a user with a custom greeting."""
        return f"{greeting}, {name}!"

    tool_instance = greet_user()
    ctx = get_test_tool_context()

    # Test with default parameter
    result = tool_instance.run(ctx, name="Alice")
    assert result == "Hello, Alice!"

    # Test with custom parameter
    result = tool_instance.run(ctx, name="Bob", greeting="Hi")
    assert result == "Hi, Bob!"


def test_tool_with_context_parameter() -> None:
    """Test tool decorator with context parameter."""

    @tool
    def get_user_id(ctx: ToolRunContext) -> str:
        """Get the current user ID from context."""
        return ctx.end_user.external_id

    tool_instance = get_user_id()
    ctx = get_test_tool_context()

    result = tool_instance.run(ctx)
    assert result == "test_user"


def test_tool_with_context_named_context() -> None:
    """Test tool decorator with context parameter named 'context'."""

    @tool
    def get_plan_id(context: ToolRunContext) -> str:
        """Get the current plan ID from context."""
        return str(context.plan_run.plan_id)

    tool_instance = get_plan_id()
    ctx = get_test_tool_context()

    result = tool_instance.run(ctx)
    assert result == str(ctx.plan_run.plan_id)


def test_tool_with_mixed_parameters() -> None:
    """Test tool decorator with both regular and context parameters."""

    @tool
    def personalized_message(message: str, ctx: ToolRunContext, prefix: str = "Message") -> str:
        """Create a personalized message for the current user."""
        return f"{prefix} for {ctx.end_user.external_id}: {message}"

    tool_instance = personalized_message()
    ctx = get_test_tool_context()

    result = tool_instance.run(ctx, message="Hello World")
    assert result == "Message for test_user: Hello World"

    result = tool_instance.run(ctx, message="Test", prefix="Alert")
    assert result == "Alert for test_user: Test"


def test_tool_with_complex_types() -> None:
    """Test tool decorator with complex type annotations."""

    @tool
    def process_data(items: list[str], count: int | None = None) -> dict[str, int]:
        """Process a list of items and return counts."""
        if count is None:
            count = len(items)
        return {"total_items": len(items), "requested_count": count}

    tool_instance = process_data()
    ctx = get_test_tool_context()

    result = tool_instance.run(ctx, items=["a", "b", "c"])
    assert result == {"total_items": 3, "requested_count": 3}

    result = tool_instance.run(ctx, items=["x", "y"], count=5)
    assert result == {"total_items": 2, "requested_count": 5}


def test_tool_raises_errors() -> None:
    """Test that decorated tools can raise Tool errors."""

    @tool
    def failing_tool(should_fail: bool, error_type: str = "soft") -> str:
        """Fail in different ways."""
        if should_fail:
            if error_type == "hard":
                raise ToolHardError("Hard error occurred")
            if error_type == "soft":
                raise ToolSoftError("Soft error occurred")
            raise ValueError("Unknown error")
        return "Success"

    tool_instance = failing_tool()
    ctx = get_test_tool_context()

    # Test successful execution
    result = tool_instance.run(ctx, should_fail=False)
    assert result == "Success"

    # Test soft error
    with pytest.raises(ToolSoftError):
        tool_instance.run(ctx, should_fail=True, error_type="soft")

    # Test hard error
    with pytest.raises(ToolHardError):
        tool_instance.run(ctx, should_fail=True, error_type="hard")


def test_weather_tool_example(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the weather tool example from the requirement."""
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "")

    @tool
    def weather_tool(city: str) -> str:
        """Retrieve the weather of the provided city."""
        # Mock implementation for testing
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key or api_key == "":
            raise ToolHardError("OPENWEATHERMAP_API_KEY is required")

        # Mock weather data for testing
        return f"The current weather in {city} is sunny with a temperature of 22°C."

    # Create tool instance
    tool_instance = weather_tool()

    # Check properties match expected format
    assert tool_instance.id == "weather_tool"
    assert tool_instance.name == "Weather Tool"
    assert tool_instance.description == "Retrieve the weather of the provided city."
    assert tool_instance.output_schema == ("str", "Output from weather_tool function")

    # Check args schema has city parameter
    schema = tool_instance.args_schema
    assert "city" in schema.model_fields
    city_field = schema.model_fields["city"]
    assert city_field.annotation is str

    # Test without API key (should raise error)
    ctx = get_test_tool_context()
    with pytest.raises(ToolHardError, match="OPENWEATHERMAP_API_KEY is required"):
        tool_instance.run(ctx, city="London")


def test_tool_class_naming() -> None:
    """Test that tool classes get proper names."""

    @tool
    def my_custom_tool(value: str) -> str:
        """Test custom tool functionality."""
        return value.upper()

    tool_instance = my_custom_tool()

    # Check the class name
    assert tool_instance.__class__.__name__ == "MyCustomToolTool"
    assert tool_instance.__class__.__qualname__ == "MyCustomToolTool"


def test_tool_validation_missing_return_type() -> None:
    """Test that tools without return type annotations are rejected."""

    def invalid_tool_func(a: int, b: int):  # type: ignore[no-untyped-def]  # noqa: ANN202
        """Invalid tool function."""
        return a + b

    with pytest.raises(ValueError, match="must have a return type annotation"):
        tool(invalid_tool_func)


def test_tool_validation_non_callable() -> None:
    """Test that non-callable objects are rejected."""
    with pytest.raises(TypeError, match="must be callable"):
        tool("not a function")  # type: ignore[arg-type]


def test_tool_args_schema_generation() -> None:
    """Test detailed args schema generation."""

    @tool
    def complex_tool(
        required_str: str,
        required_float: float,
        optional_int: int = 42,
        optional_bool: bool = True,
    ) -> str:
        """Tool with various parameter types."""
        return f"{required_str}-{optional_int}-{optional_bool}-{required_float}"

    tool_instance = complex_tool()
    schema = tool_instance.args_schema

    # Check required fields
    assert "required_str" in schema.model_fields
    assert "required_float" in schema.model_fields

    # Check optional fields with defaults
    assert "optional_int" in schema.model_fields
    assert "optional_bool" in schema.model_fields

    # Test schema instantiation
    schema_instance = schema(
        required_str="test",
        required_float=3.14,
    )
    assert schema_instance.required_str == "test"  # pyright: ignore[reportAttributeAccessIssue]
    assert schema_instance.optional_int == 42  # pyright: ignore[reportAttributeAccessIssue]
    assert schema_instance.optional_bool is True  # pyright: ignore[reportAttributeAccessIssue]
    assert schema_instance.required_float == 3.14  # pyright: ignore[reportAttributeAccessIssue]


def test_tool_to_langchain() -> None:
    """Test that decorated tools can be converted to LangChain tools."""

    @tool
    def simple_tool(text: str) -> str:
        """Convert text to uppercase for LangChain testing."""
        return text.upper()

    tool_instance = simple_tool()
    ctx = get_test_tool_context()

    # Convert to LangChain tool
    lc_tool = tool_instance.to_langchain(ctx)

    # Check properties
    assert lc_tool.name == "Simple_Tool"
    assert "convert text to uppercase for langchain testing" in lc_tool.description.lower()
    assert lc_tool.args_schema == tool_instance.args_schema


def test_tool_serialization() -> None:
    """Test that decorated tools can be serialized."""

    @tool
    def serializable_tool(data: str) -> str:
        """Tool that can be serialized."""
        return data

    tool_instance = serializable_tool()

    # Test string representation
    str_repr = str(tool_instance)
    assert "serializable_tool" in str_repr
    assert "Serializable Tool" in str_repr

    # Test JSON serialization
    json_data = tool_instance.model_dump_json()
    assert isinstance(json_data, str)
    assert "serializable_tool" in json_data


def test_annotated_string_description() -> None:
    """Test Annotated[Type, "description"] pattern."""

    @tool
    def say_hello(
        name: Annotated[str, "The name of the person to say hello to"],
    ) -> str:
        """Say hello to someone."""
        return f"Hello, {name}!"

    tool_instance = say_hello()

    # Check that the tool was created properly
    assert tool_instance.id == "say_hello"
    assert tool_instance.name == "Say Hello"
    assert tool_instance.description == "Say hello to someone."

    # Check the args schema
    schema = tool_instance.args_schema
    assert "name" in schema.model_fields
    name_field = schema.model_fields["name"]
    assert name_field.annotation is str
    assert name_field.description == "The name of the person to say hello to"

    # Test execution
    ctx = get_test_tool_context()
    result = tool_instance.run(ctx, name="Alice")
    assert result == "Hello, Alice!"


def test_annotated_field_description() -> None:
    """Test Annotated[Type, Field(description="...")] pattern."""

    @tool
    def calculate_area(
        length: Annotated[float, Field(description="The length of the rectangle")],
        width: Annotated[float, Field(description="The width of the rectangle", gt=0)],
    ) -> float:
        """Calculate the area of a rectangle."""
        return length * width

    tool_instance = calculate_area()

    # Check that the tool was created properly
    assert tool_instance.id == "calculate_area"
    assert tool_instance.name == "Calculate Area"

    # Check the args schema
    schema = tool_instance.args_schema
    assert "length" in schema.model_fields
    assert "width" in schema.model_fields

    length_field = schema.model_fields["length"]
    width_field = schema.model_fields["width"]

    assert length_field.annotation is float
    assert length_field.description == "The length of the rectangle"

    assert width_field.annotation is float
    assert width_field.description == "The width of the rectangle"

    # Test execution
    ctx = get_test_tool_context()
    result = tool_instance.run(ctx, length=5.0, width=3.0)
    assert result == 15.0


def test_mixed_annotation_patterns() -> None:
    """Test mixing different annotation patterns."""

    @tool
    def mixed_function(
        required_annotated: Annotated[str, "A required parameter with annotation"],
        required_regular: int,
        optional_annotated: Annotated[
            str, Field(description="An optional parameter", min_length=1)
        ] = "default",
        optional_regular: bool = True,
    ) -> str:
        """Test function with mixed annotation patterns."""
        return f"{required_annotated}-{required_regular}-{optional_annotated}-{optional_regular}"

    tool_instance = mixed_function()

    # Check the args schema
    schema = tool_instance.args_schema
    fields = schema.model_fields

    # Check required_annotated
    assert "required_annotated" in fields
    assert fields["required_annotated"].annotation is str
    assert fields["required_annotated"].description == "A required parameter with annotation"

    # Check required_regular (should get fallback description)
    assert "required_regular" in fields
    assert fields["required_regular"].annotation is int
    description = fields["required_regular"].description
    assert description is not None
    assert "Parameter required_regular for mixed_function" in description

    # Check optional_annotated
    assert "optional_annotated" in fields
    assert fields["optional_annotated"].annotation is str
    assert fields["optional_annotated"].description == "An optional parameter"
    assert fields["optional_annotated"].default == "default"
    # Validate that min_length constraint is present
    optional_field = fields["optional_annotated"]
    # Check if the min_length constraint is in the field metadata
    assert hasattr(optional_field, "metadata")
    # Extract field constraints from metadata - look for MinLen object
    min_len_constraint = None
    for meta in optional_field.metadata:
        if hasattr(meta, "min_length") and str(type(meta).__name__) == "MinLen":
            min_len_constraint = meta
            break
    assert min_len_constraint is not None
    assert min_len_constraint.min_length == 1

    # Test that the schema rejects empty strings (min_length=1 constraint)
    with pytest.raises(ValueError, match="String should have at least 1 character"):
        schema(required_annotated="test", required_regular=42, optional_annotated="")

    # Check optional_regular
    assert "optional_regular" in fields
    assert fields["optional_regular"].annotation is bool
    assert fields["optional_regular"].default is True

    # Test execution
    ctx = get_test_tool_context()
    result = tool_instance.run(ctx, required_annotated="test", required_regular=42)
    assert result == "test-42-default-True"


def test_get_type_hints_exception_handling() -> None:
    """Test exception handling in _create_args_schema when get_type_hints fails."""
    import inspect

    # Create a function that will cause get_type_hints to fail
    def problematic_function(
        param: "NonExistentType",  # pyright: ignore[reportUndefinedVariable]  # noqa: F821 ARG001
    ) -> str:  # Forward reference to non-existent type
        return "test"

    sig = inspect.signature(problematic_function)

    schema = _create_args_schema(sig, "test_func", problematic_function)

    # Should still create a valid schema
    assert issubclass(schema, BaseModel)
    assert "param" in schema.model_fields


def test_empty_parameter_annotation_fallback() -> None:
    """Test fallback when parameter annotation is empty."""
    import inspect

    # Create a function with no type annotation
    def test_func(param) -> str:  # No annotation  # noqa: ANN001 ARG001
        return "test"

    sig = inspect.signature(test_func)

    schema = _create_args_schema(sig, "test_func", test_func)

    # Should still create a valid schema with Any type
    assert issubclass(schema, BaseModel)
    assert "param" in schema.model_fields

    # The parameter should have type Any (since it had no annotation)
    param_field = schema.model_fields["param"]
    assert param_field.annotation is Any


def test_malformed_annotated_type() -> None:
    """Test handling of malformed Annotated types."""
    import inspect

    # Create a mock malformed Annotated type (has origin but no args)
    class MockAnnotated:
        pass

    # Mock get_origin and get_args to simulate malformed Annotated
    with (
        unittest.mock.patch("portia.tool_decorator.get_origin", return_value=Annotated),
        unittest.mock.patch("portia.tool_decorator.get_args", return_value=[]),
    ):
        param = inspect.Parameter("test_param", inspect.Parameter.POSITIONAL_OR_KEYWORD)

        param_type, field_info = _extract_type_and_field_info(
            MockAnnotated, param, "test_param", "test_func"
        )

        assert param_type is Any
        assert field_info.description is not None
        assert "Parameter test_param for test_func" in field_info.description


def test_field_with_custom_default() -> None:
    """Test Field with custom default value handling."""

    @tool
    def tool_with_field_default(
        name: Annotated[
            str, Field(default="default_name", description="Name parameter")
        ] = "default_name_foo",
    ) -> str:
        """Tool with Field default."""
        return f"Hello, {name}!"

    tool_instance = tool_with_field_default()

    # Check the args schema
    schema = tool_instance.args_schema
    name_field = schema.model_fields["name"]
    assert name_field.default == "default_name_foo"
    assert name_field.description == "Name parameter"

    # Test execution with default
    ctx = get_test_tool_context()
    result = tool_instance.run(ctx)
    assert result == "Hello, default_name_foo!"

    # Test execution with custom value
    result = tool_instance.run(ctx, name="Alice")
    assert result == "Hello, Alice!"


def test_tool_with_invalid_annotation_metadata() -> None:
    """Test that invalid annotation metadata raises an error."""
    with pytest.raises(ValueError, match="Unsupported annotation metadata: 123"):

        @tool
        def tool_with_invalid_annotation_metadata(
            name: Annotated[str, 123],
        ) -> str:
            """Tool with invalid annotation metadata."""
            return f"Hello, {name}!"


def test_tool_description_length_validation() -> None:
    """Test that tool descriptions exceeding MAX_TOOL_DESCRIPTION_LENGTH raise error."""

    def tool_with_long_description() -> str:
        return "result"

    tool_with_long_description.__doc__ = "x" * 16385
    tool_class = tool(tool_with_long_description)

    # The error should be raised when we instantiate the tool
    with pytest.raises(InvalidToolDescriptionError):
        tool_class()


def test_tool_with_context_parameter_name_invalid() -> None:
    """Test that tool with context parameter name invalid raises an error."""
    with pytest.raises(
        ValueError,
        match="Tool tool_with_context_parameter_name_invalid has a ToolRunContext parameter that "
        "is not named 'ctx' or 'context'. This is not allowed as it causes errors when the tool "
        "inputs are validated.",
    ):

        @tool
        def tool_with_context_parameter_name_invalid(
            tool_context: ToolRunContext,  # noqa: ARG001
            arg1: str,  # noqa: ARG001
        ) -> str:
            """Tool with context parameter name invalid."""
            return "test"
