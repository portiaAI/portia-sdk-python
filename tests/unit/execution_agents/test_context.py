"""test context."""

from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl

from portia.clarification import ActionClarification, InputClarification
from portia.end_user import EndUser
from portia.execution_agents.context import StepInput, build_context
from portia.execution_agents.output import LocalDataValue, Output
from portia.plan import Variable
from portia.tool import ToolRunContext
from tests.utils import get_test_config, get_test_plan_run


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
        "$email_body": LocalDataValue(value="The body of the email"),
        "$email_title": LocalDataValue(value="Example email"),
        "$email_address": LocalDataValue(value="test@example.com"),
        "$london_weather": LocalDataValue(value="rainy"),
    }


def test_context_empty() -> None:
    """Test that the context is set up correctly."""
    (plan, plan_run) = get_test_plan_run()
    plan_run.outputs.step_outputs = {}
    context = build_context(
        ToolRunContext(
            end_user=EndUser(
                external_id="123",
                additional_data={"email": "hello@world.com"},
            ),
            plan_run=plan_run,
            plan=plan,
            config=get_test_config(),
            clarifications=[],
        ),
        plan_run,
        [],
    )
    assert "System Context:" in context
    assert len(context) == 42  # length should always be the same


def test_context_execution_context() -> None:
    """Test that the context is set up correctly."""
    (plan, plan_run) = get_test_plan_run()

    context = build_context(
        ToolRunContext(
            end_user=EndUser(
                external_id="123",
                additional_data={"email": "hello@world.com"},
            ),
            plan_run=plan_run,
            plan=plan,
            config=get_test_config(),
            clarifications=[],
        ),
        plan_run,
        [StepInput(name="$output1", value="test1", description="Previous output 1")],
    )
    assert "System Context:" in context
    assert "user_id" in context
    assert "123" in context
    assert "test1" in context


def test_context_inputs_and_outputs(inputs: list[Variable], outputs: dict[str, Output]) -> None:
    """Test that the context is set up correctly with inputs and outputs."""
    (plan, plan_run) = get_test_plan_run()
    plan.steps[0].inputs = inputs
    plan_run.outputs.step_outputs = outputs
    context = build_context(
        ToolRunContext(
            end_user=EndUser(
                external_id="123",
                additional_data={"email": "hello@world.com"},
            ),
            plan_run=plan_run,
            plan=plan,
            config=get_test_config(),
            clarifications=[],
        ),
        plan_run,
        [],
    )
    for variable in inputs:
        assert variable.name in context
    for name, output in outputs.items():
        assert name in context
        if output.get_value():
            val = output.get_value()
            assert isinstance(val, str)
            assert val in context


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
            source="Test execution agents context",
        ),
        InputClarification(
            plan_run_id=plan_run.id,
            argument_name="$email_cc",
            user_guidance="email cc list",
            response="bob@bla.com",
            step=1,
            source="Test execution agents context",
        ),
        ActionClarification(
            plan_run_id=plan_run.id,
            action_url=HttpUrl("http://example.com"),
            user_guidance="click on the link",
            source="Test execution agents context",
        ),
    ]
    plan_run.outputs.clarifications = clarifications
    context = build_context(
        ToolRunContext(
            end_user=EndUser(
                external_id="123",
                additional_data={"email": "hello@world.com"},
            ),
            plan=plan,
            plan_run=plan_run,
            config=get_test_config(),
            clarifications=clarifications,
        ),
        plan_run,
        [
            StepInput(
                name="$email_address",
                value="test@example.com",
                description="Target recipient for email",
            ),
            StepInput(
                name="$email_body",
                value="The body of the email",
                description="Content for email",
            ),
            StepInput(
                name="$email_title",
                value="Example email",
                description="Title for email",
            ),
        ],
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
Details on the end user.
You can use this information when the user mentions themselves (i.e send me an email)
if no other information is provided in the task.
end_user_id:123
end_user_name:
end_user_email:
end_user_phone:
end_user_attributes:{{"email": "hello@world.com"}}
----------
System Context:
Today's date is {datetime.now(UTC).strftime('%Y-%m-%d')}"""
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
            source="Test execution agents context",
        ),
        ActionClarification(
            plan_run_id=plan_run.id,
            action_url=HttpUrl("http://example.com"),
            user_guidance="click on the link",
            step=1,
            source="Test execution agents context",
        ),
    ]
    plan.steps[0].inputs = inputs
    plan_run.outputs.step_outputs = outputs
    plan_run.outputs.clarifications = clarifications
    context = build_context(
        ToolRunContext(
            end_user=EndUser(external_id="123"),
            plan=plan,
            plan_run=plan_run,
            config=get_test_config(),
            clarifications=clarifications,
        ),
        plan_run,
        [],
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
