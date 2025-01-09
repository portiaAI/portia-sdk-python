"""test context."""

import pytest
from pydantic import HttpUrl

from portia.agents.base_agent import Output
from portia.agents.context import build_context
from portia.clarification import ActionClarification, InputClarification
from portia.plan import Variable


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
def outputs() -> dict[str, Output]:
    """Return a dictionary of outputs for pytest fixtures."""
    return {
        "$email_body": Output(value="The body of the email"),
        "$london_weather": Output(value="rainy"),
    }


def test_context_empty() -> None:
    """Test that the context is set up correctly."""
    context = build_context(
        [],
        {},
        [],
        [],
    )
    assert "System Context:" in context
    assert len(context) == 42  # length should always be the same


def test_context_inputs_only(inputs: list[Variable]) -> None:
    """Test that the context is set up correctly with inputs."""
    context = build_context(inputs, {}, [])
    for variable in inputs:
        if variable.value:
            assert variable.value in context


def test_context_inputs_and_outputs(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with inputs and outputs."""
    context = build_context(inputs, outputs, [])
    for variable in inputs:
        if variable.value:
            assert variable.value in context
    for name, output in outputs.items():
        assert name in context
        if output.value:
            assert output.value in context


def test_system_context() -> None:
    """Test that the system context is set up correctly."""
    context = build_context([], {}, [], ["system context 1", "system context 2"])
    assert "system context 1" in context
    assert "system context 2" in context


def test_all_contexts(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with all contexts."""
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
    context = build_context(
        inputs,
        outputs,
        clarifications,
        ["system context 1", "system context 2"],
    )
    assert (
        context
        == """Additional context: You MUST use this information to complete your task.
Inputs: the original inputs provided by the planner
input_name: $email_address
input_value: test@example.com
input_description: Target recipient for email
----------
input_name: $email_body
input_value: value='The body of the email'
input_description: Content for email
----------
input_name: $email_title
input_value: Example email
input_description: Title for email
----------
Broader context: This may be useful information from previous steps that can indirectly help you.
output_name: $london_weather
output_value: value='rainy'
----------
Clarifications:
This section contains the user provided response to previous clarifications
They should take priority over any other context given.
input_name: $email_cc
clarification_reason: email cc list
input_value: bob@bla.com
----------
System Context:
Today's date is 2025-01-09
system context 1
system context 2"""
    )


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
    for variable in inputs:
        if variable.value:
            assert variable.value in context
    for name, output in outputs.items():
        assert name in context
        if output.value:
            assert output.value in context
    assert "email cc list" in context
    assert "bob@bla.com" in context
