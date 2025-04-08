"""test context."""

from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl

from portia.clarification import ActionClarification, InputClarification
from portia.execution_agents.context import build_context
from portia.execution_agents.output import LocalOutput, Output
from portia.execution_context import ExecutionContext
from portia.plan import Step, Variable
from tests.utils import get_test_plan_run


@pytest.fixture
def inputs() -> list[Variable]:
    """Return a list of inputs for pytest fixtures."""
    return [
        Variable(
            name="$email_address",
            description="Target recipient for email",
        ),
        Variable(name="$email_body", description="Content for email"),
        Variable(name="$email_title", description="Title for email"),
    ]


@pytest.fixture
def outputs() -> dict[str, Output]:
    """Return a dictionary of outputs for pytest fixtures."""
    return {
        "$email_body": LocalOutput(value="The body of the email"),
        "$email_title": LocalOutput(value="Example email"),
        "$email_address": LocalOutput(value="test@example.com"),
        "$london_weather": LocalOutput(value="rainy"),
    }


def test_context_empty() -> None:
    """Test that the context is set up correctly."""
    (_, plan_run) = get_test_plan_run()
    plan_run.outputs.step_outputs = {}
    context = build_context(
        ExecutionContext(),
        Step(inputs=[], output="", task=""),
        plan_run,
        {},
    )
    assert "System Context:" in context
    assert len(context) == 42  # length should always be the same


def test_context_execution_context() -> None:
    """Test that the context is set up correctly."""
    (plan, plan_run) = get_test_plan_run()

    context = build_context(
        ExecutionContext(additional_data={"user_id": "123"}),
        plan.steps[0],
        plan_run,
        {"test_output": LocalOutput(value="test_value")},
    )
    assert "System Context:" in context
    assert "user_id" in context
    assert "123" in context


def test_context_inputs_and_outputs(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with inputs and outputs."""
    (plan, plan_run) = get_test_plan_run()
    plan.steps[0].inputs = inputs
    plan_run.outputs.step_outputs = outputs
    context = build_context(
        ExecutionContext(),
        plan.steps[0],
        plan_run,
        plan_run.outputs.step_outputs,
    )
    for variable in inputs:
        assert variable.name in context
    for name, output in outputs.items():
        assert name in context
        if output.get_value():
            val = output.get_value()
            assert isinstance(val, str)
            assert val in context


def test_system_context() -> None:
    """Test that the system context is set up correctly."""
    (plan, plan_run) = get_test_plan_run()
    context = build_context(
        ExecutionContext(
            execution_agent_system_context_extension=["system context 1", "system context 2"],
        ),
        plan.steps[0],
        plan_run,
        plan_run.outputs.step_outputs,
    )
    assert "system context 1" in context
    assert "system context 2" in context


def test_all_contexts(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with all contexts."""
    (plan, plan_run) = get_test_plan_run()
    plan.steps[0].inputs = inputs
    plan_run.outputs.step_outputs = outputs
    clarifications = [
        InputClarification(
            plan_run_id=plan_run.id,
            argument_name="$email_cc",
            user_guidance="email cc list",
            response="bob@bla.com",
            step=0,
        ),
        InputClarification(
            plan_run_id=plan_run.id,
            argument_name="$email_cc",
            user_guidance="email cc list",
            response="bob@bla.com",
            step=1,
        ),
        ActionClarification(
            plan_run_id=plan_run.id,
            action_url=HttpUrl("http://example.com"),
            user_guidance="click on the link",
        ),
    ]
    plan_run.outputs.clarifications = clarifications
    context = build_context(
        ExecutionContext(
            execution_agent_system_context_extension=["system context 1", "system context 2"],
            end_user_id="123",
            additional_data={"email": "hello@world.com"},
        ),
        plan.steps[0],
        plan_run,
        plan_run.outputs.step_outputs,
    )
    # as LLMs are sensitive even to white space formatting we do a complete match here
    assert (
        context
        == f"""Additional context: You MUST use this information to complete your task.
Inputs: the original inputs provided by the planning_agent
input_name: $email_address
input_value: test@example.com
input_description: Target recipient for email
----------
input_name: $email_body
input_value: The body of the email
input_description: Content for email
----------
input_name: $email_title
input_value: Example email
input_description: Title for email
----------
Broader context: This may be useful information from previous steps that can indirectly help you.
output_name: $london_weather
output_value: rainy
----------
Clarifications:
This section contains the user provided response to previous clarifications
for the current step. They should take priority over any other context given.
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
    (plan, plan_run) = get_test_plan_run()
    clarifications = [
        InputClarification(
            plan_run_id=plan_run.id,
            argument_name="$email_cc",
            user_guidance="email cc list",
            response="bob@bla.com",
            step=0,
        ),
        ActionClarification(
            plan_run_id=plan_run.id,
            action_url=HttpUrl("http://example.com"),
            user_guidance="click on the link",
            step=1,
        ),
    ]
    plan.steps[0].inputs = inputs
    plan_run.outputs.step_outputs = outputs
    plan_run.outputs.clarifications = clarifications
    context = build_context(
        ExecutionContext(
            execution_agent_system_context_extension=["system context 1", "system context 2"],
        ),
        plan.steps[0],
        plan_run,
        plan_run.outputs.step_outputs,
    )
    for variable in inputs:
        assert variable.name in context
    for name, output in outputs.items():
        assert name in context
        if output.get_value():
            val = output.get_value()
            assert isinstance(val, str)
            assert val in context
    assert "email cc list" in context
    assert "bob@bla.com" in context
