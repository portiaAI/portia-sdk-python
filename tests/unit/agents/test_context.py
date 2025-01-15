"""test context."""

from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl

from portia.agents.base_agent import Output
from portia.agents.context import build_context
from portia.clarification import ActionClarification, InputClarification
from portia.context import ExecutionContext
from portia.plan import Step, Variable
from tests.utils import get_test_workflow


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
    (_, workflow) = get_test_workflow()
    context = build_context(
        ExecutionContext(),
        Step(inputs=[], output="", task=""),
        workflow,
    )
    assert "System Context:" in context
    assert len(context) == 42  # length should always be the same


def test_context_execution_context() -> None:
    """Test that the context is set up correctly."""
    (plan, workflow) = get_test_workflow()
    context = build_context(
        ExecutionContext(additional_data={"user_id": "123"}),
        plan.steps[0],
        workflow,
    )
    assert "System Context:" in context
    assert "user_id" in context
    assert "123" in context


def test_context_inputs_only(inputs: list[Variable]) -> None:
    """Test that the context is set up correctly with inputs."""
    (plan, workflow) = get_test_workflow()
    plan.steps[0].inputs = inputs
    context = build_context(
        ExecutionContext(),
        plan.steps[0],
        workflow,
    )
    for variable in inputs:
        if variable.value:
            assert variable.value in context


def test_context_inputs_and_outputs(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with inputs and outputs."""
    (plan, workflow) = get_test_workflow()
    plan.steps[0].inputs = inputs
    workflow.outputs.step_outputs = outputs
    context = build_context(
        ExecutionContext(),
        plan.steps[0],
        workflow,
    )
    for variable in inputs:
        if variable.value:
            assert variable.value in context
    for name, output in outputs.items():
        assert name in context
        if output.value:
            assert output.value in context


def test_system_context() -> None:
    """Test that the system context is set up correctly."""
    (plan, workflow) = get_test_workflow()
    context = build_context(
        ExecutionContext(agent_system_context_extension=["system context 1", "system context 2"]),
        plan.steps[0],
        workflow,
    )
    assert "system context 1" in context
    assert "system context 2" in context


def test_all_contexts(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with all contexts."""
    (plan, workflow) = get_test_workflow()
    plan.steps[0].inputs = inputs
    workflow.outputs.step_outputs = outputs
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
    workflow.outputs.clarifications = clarifications
    context = build_context(
        ExecutionContext(
            agent_system_context_extension=["system context 1", "system context 2"],
            end_user_id="123",
            additional_data={"email": "hello@world.com"},
        ),
        plan.steps[0],
        workflow,
    )
    # as LLMs are sensitive even to white space formatting we do a complete match here
    assert (
        context
        == f"""Additional context: You MUST use this information to complete your task.
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
Metadata: This section contains general context about this execution.
end_user_id: 123
context_key_name: email context_key_value: hello@world.com
----------
System Context:
Today's date is {datetime.now(UTC).strftime('%Y-%m-%d')}
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
    (plan, workflow) = get_test_workflow()
    plan.steps[0].inputs = inputs
    workflow.outputs.step_outputs = outputs
    workflow.outputs.clarifications = clarifications
    context = build_context(
        ExecutionContext(agent_system_context_extension=["system context 1", "system context 2"]),
        plan.steps[0],
        workflow,
    )
    for variable in inputs:
        if variable.value:
            assert variable.value in context
    for name, output in outputs.items():
        assert name in context
        if output.value:
            assert output.value in context
    assert "email cc list" in context
    assert "bob@bla.com" in context
