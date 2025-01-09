"""test context."""

import pytest
from pydantic import HttpUrl

from portia.agents.base_agent import Output
from portia.agents.context import build_context
from portia.clarification import ActionClarification, InputClarification
from portia.plan import Variable


def test_context_empty() -> None:
    """Test that the context is set up correctly."""
    context = build_context([], {}, [])
    assert context == "No additional context", "Expected no additional context."


@pytest.fixture
def inputs() -> list[Variable]:
    """Return a list of inputs for pytest fixtures."""
    return [
        Variable(
            name="$email_address",
            value="test@example.com",
            description="Target recipient for email",
        ),
        Variable(name="$email_body", description="Content for email"),
        Variable(name="$email_title", value="Example email", description="Title for email"),
    ]


@pytest.fixture
def outputs() -> dict[str, str]:
    """Return a dictionary of outputs for pytest fixtures."""
    return {"$email_body": "The body of the email", "$london_weather": "rainy"}


def test_context_inputs_only(inputs: list[Variable]) -> None:
    """Test that the context is set up correctly with inputs."""
    context = build_context(inputs, {}, [])
    expected_output = """Additional context: You MUST use this information to complete your task.
name: $email_address
value: test@example.com
description: Target recipient for email

----------

name: $email_title
value: Example email
description: Title for email

----------

"""
    assert context == expected_output


def test_context_inputs_and_outputs(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with inputs and outputs."""
    context = build_context(inputs, outputs, [])
    expected_output = """Additional context: You MUST use this information to complete your task.
name: $email_address
value: test@example.com
description: Target recipient for email

----------

name: $email_body
value: The body of the email
description: Content for email

----------

name: $email_title
value: Example email
description: Title for email

----------


Broader context: This may be useful information from previous steps that can indirectly help you.
name: $london_weather
value: rainy

----------

"""
    assert context == expected_output


def test_system_context() -> None:
    """Test that the system context is set up correctly."""
    context = build_context([], {}, [], system_context=["system context 1", "system context 2"])
    assert (
        context
        == """Additional context: You MUST use this information to complete your task.

System Context:
system context 1
system context 2

----------

"""
    )


def test_system_context_empty() -> None:
    """Test that the system context is set up correctly when empty."""
    context = build_context([], {}, [], system_context=[])
    assert context == "No additional context"


def test_all_contexts(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with all contexts."""
    context = build_context(
        inputs,
        outputs,
        [],
        system_context=["system context 1", "system context 2"],
    )
    expected_context = """Additional context: You MUST use this information to complete your task.
name: $email_address
value: test@example.com
description: Target recipient for email

----------

name: $email_body
value: The body of the email
description: Content for email

----------

name: $email_title
value: Example email
description: Title for email

----------


Broader context: This may be useful information from previous steps that can indirectly help you.
name: $london_weather
value: rainy

----------


System Context:
system context 1
system context 2

----------

"""
    assert context == expected_context


def test_context_inputs_outputs_clarifications(
    inputs: list[Variable],
    outputs: dict[str, Output],
) -> None:
    """Test that the context is set up correctly with inputs, outputs, and missing args."""
    clarifications = [
        InputClarification(
            argument_name="$email_cc",
            user_guidance="email cc list",
            response="bob@bla.com",
        ),
        ActionClarification(
            action_url=HttpUrl("http://example.com"),
            user_guidance="click on the link",
        ),
    ]
    context = build_context(inputs, outputs, clarifications)
    expected_context = """Additional context: You MUST use this information to complete your task.
name: $email_address
value: test@example.com
description: Target recipient for email

----------

name: $email_body
value: The body of the email
description: Content for email

----------

name: $email_title
value: Example email
description: Title for email

----------

Clarifications: This section contains user provided clarifications that might be useful to complete your task.
argument: $email_cc
clarification reason: email cc list
value: bob@bla.com

----------


Broader context: This may be useful information from previous steps that can indirectly help you.
name: $london_weather
value: rainy

----------

"""  # noqa: E501
    assert context == expected_context
