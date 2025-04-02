from typing import List, Optional

from pydantic import BaseModel

from portia.tool_decorator import tool


def test_simple_function_decorator():
    """Test basic function decoration with simple types."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together.

        Args:
            a: First number
            b: Second number

        Returns:
            int: Sum of a and b

        """
        return a + b

    # Check basic attributes
    assert add.id == "add"
    assert add.name == "add"
    assert "Add two integers together" in add.description

    # Check input schema
    input_schema = add.args_schema
    assert issubclass(input_schema, BaseModel)
    assert "a" in input_schema.model_fields
    assert "b" in input_schema.model_fields
    assert input_schema.model_fields["a"].annotation == int
    assert input_schema.model_fields["b"].annotation == int

    # Check output schema
    assert add.output_schema[0] == "int"
    assert "Sum of a and b" in add.output_schema[1]


def test_complex_types_decorator():
    """Test function decoration with complex types and optional parameters."""

    class User(BaseModel):
        name: str
        age: int

    @tool
    def process_users(users: List[User], filter_age: Optional[int] = None) -> List[str]:
        """Process a list of users and return their names.

        Args:
            users: List of user objects to process
            filter_age: Optional age filter

        Returns:
            List[str]: List of user names

        """
        return [user.name for user in users]

    # Check basic attributes
    assert process_users.id == "process_users"
    assert process_users.name == "process_users"
    assert "Process a list of users" in process_users.description

    # Check input schema
    input_schema = process_users.args_schema
    assert issubclass(input_schema, BaseModel)
    assert "users" in input_schema.model_fields
    assert "filter_age" in input_schema.model_fields

    # Check field types - just verify it's a List of User objects
    users_type = str(input_schema.model_fields["users"].annotation)
    assert users_type.startswith("typing.List[")
    assert "User" in users_type
    assert users_type.endswith("]")

    filter_type = str(input_schema.model_fields["filter_age"].annotation)
    assert filter_type.startswith("typing.Optional[")
    assert "int" in filter_type.lower()
    assert filter_type.endswith("]")

    # Check output schema
    assert process_users.output_schema[0] == "List"
    assert "List of user names" in process_users.output_schema[1]


def test_class_method_decorator():
    """Test decorator with class methods."""

    class Calculator:
        @tool
        def multiply(self, x: float, y: float) -> float:
            """Multiply two numbers.

            Args:
                x: First number
                y: Second number

            Returns:
                float: Product of x and y

            """
            return x * y

    # Check that self parameter is excluded from schema
    input_schema = Calculator.multiply.args_schema
    assert "self" not in input_schema.model_fields
    assert "x" in input_schema.model_fields
    assert "y" in input_schema.model_fields


def test_no_type_hints_decorator():
    """Test decorator with function lacking type hints."""

    @tool
    def greet(name, message="Hello"):
        """Greet someone with a message.

        Args:
            name: Name of person to greet
            message: Greeting message

        Returns:
            str: Complete greeting

        """
        return f"{message}, {name}!"

    # Check that Any is used as default type
    input_schema = greet.args_schema
    assert str(input_schema.model_fields["name"].annotation) == "typing.Any"
    assert str(input_schema.model_fields["message"].annotation) == "typing.Any"

    # Check output schema
    assert greet.output_schema[0] == "Any"
