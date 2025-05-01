"""Tests for the DefaultIntrospectionAgent module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from portia.config import GenerativeModelsConfig
from portia.execution_agents.output import LocalDataValue
from portia.introspection_agents.default_introspection_agent import DefaultIntrospectionAgent
from portia.introspection_agents.introspection_agent import (
    BaseIntrospectionAgent,
    PreStepIntrospection,
    PreStepIntrospectionOutcome,
)
from portia.model import GenerativeModel, Message
from portia.plan import Plan, PlanContext, Step, Variable
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState
from portia.prefixed_uuid import PlanUUID
from portia.storage import InMemoryStorage
from tests.utils import get_test_config


@pytest.fixture
def mock_introspection_model() -> MagicMock:
    """Mock GenerativeModel object for testing."""
    return MagicMock(spec=GenerativeModel)


@pytest.fixture
def introspection_agent(mock_introspection_model: MagicMock) -> DefaultIntrospectionAgent:
    """Create an instance of the DefaultIntrospectionAgent with mocked config."""
    mock_config = get_test_config(
        models=GenerativeModelsConfig(
            introspection_model=mock_introspection_model,
        ),
    )
    return DefaultIntrospectionAgent(config=mock_config, agent_memory=InMemoryStorage())


@pytest.fixture
def mock_plan() -> Plan:
    """Create a mock Plan for testing."""
    return Plan(
        plan_context=PlanContext(
            query="test query",
            tool_ids=["test_tool_1", "test_tool_2", "test_tool_3"],
        ),
        steps=[
            Step(
                task="Task 1",
                tool_id="test_tool_1",
                inputs=[],
                output="$result1",
            ),
            Step(
                task="Task 2",
                tool_id="test_tool_2",
                inputs=[
                    Variable(name="$result1", description="Result of task 1"),
                ],
                output="$result2",
                condition="$result1 != 'SKIPPED'",
            ),
            Step(
                task="Task 3",
                tool_id="test_tool_3",
                inputs=[
                    Variable(name="$result2", description="Result of task 2"),
                ],
                output="$final_result",
                condition="$result2 != 'SKIPPED'",
            ),
        ],
    )


@pytest.fixture
def mock_plan_run() -> PlanRun:
    """Create a mock PlanRun for testing."""
    return PlanRun(
        plan_id=PlanUUID(),
        current_step_index=1,
        state=PlanRunState.IN_PROGRESS,
        end_user_id="123",
        outputs=PlanRunOutputs(
            step_outputs={
                "$result1": LocalDataValue(value="Task 1 result", summary="Task 1 summary"),
            },
            final_output=None,
        ),
        plan_run_inputs={
            "$plan_run_input": "plan_run_input_value",
        },
    )


def test_base_introspection_agent_initialization() -> None:
    """Test BaseIntrospectionAgent initialization and default behavior."""

    # Create a minimal implementation of BaseIntrospectionAgent for testing
    class TestIntrospectionAgent(BaseIntrospectionAgent):
        """Test implementation of BaseIntrospectionAgent."""

        def pre_step_introspection(
            self,
            plan: Plan,  # noqa: ARG002
            plan_run: PlanRun,  # noqa: ARG002
        ) -> PreStepIntrospection:
            """Implement required method to test the base class."""
            return PreStepIntrospection(
                outcome=PreStepIntrospectionOutcome.CONTINUE,
                reason="Default implementation test",
            )

    config = get_test_config()
    agent_memory = InMemoryStorage()
    agent = TestIntrospectionAgent(config, agent_memory)

    assert agent.config == config
    assert agent.agent_memory == agent_memory

    empty_plan = Plan(
        plan_context=PlanContext(query="test", tool_ids=[]),
        steps=[],
    )
    empty_plan_run = PlanRun(
        plan_id=empty_plan.id,
        end_user_id="123",
    )

    result = agent.pre_step_introspection(empty_plan, empty_plan_run)

    assert isinstance(result, PreStepIntrospection)
    assert result.outcome == PreStepIntrospectionOutcome.CONTINUE
    assert result.reason == "Default implementation test"


def test_base_introspection_agent_abstract_method_raises_error() -> None:
    """Test that non-implemented pre_step_introspection raises NotImplementedError."""

    class IncompleteIntrospectionAgent(BaseIntrospectionAgent):
        """Test implementation that doesn't override the abstract method."""

        # Implement the method but have it call the parent's implementation
        def pre_step_introspection(
            self,
            plan: Plan,
            plan_run: PlanRun,
        ) -> PreStepIntrospection:
            """Call the parent's implementation which should raise NotImplementedError."""
            return super().pre_step_introspection(plan, plan_run)  # type: ignore  # noqa: PGH003

    config = get_test_config()
    agent = IncompleteIntrospectionAgent(config, InMemoryStorage())

    empty_plan = Plan(
        plan_context=PlanContext(query="test", tool_ids=[]),
        steps=[],
    )
    empty_plan_run = PlanRun(
        plan_id=empty_plan.id,
        end_user_id="123",
    )

    with pytest.raises(NotImplementedError, match="pre_step_introspection is not implemented"):
        agent.pre_step_introspection(empty_plan, empty_plan_run)


def test_pre_step_introspection_continue(
    introspection_agent: DefaultIntrospectionAgent,
    mock_introspection_model: MagicMock,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection returns CONTINUE when conditions are met."""
    # Mock the Model response response to simulate a CONTINUE outcome
    mock_introspection_model.get_structured_response.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.CONTINUE,
        reason="All conditions are met.",
    )
    result = introspection_agent.pre_step_introspection(
        plan=mock_plan,
        plan_run=mock_plan_run,
    )

    assert result.outcome == PreStepIntrospectionOutcome.CONTINUE
    assert result.reason == "All conditions are met."


def test_pre_step_introspection_skip(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
    mock_introspection_model: MagicMock,
) -> None:
    """Test pre_step_introspection returns SKIP when condition is false."""
    mock_introspection_model.get_structured_response.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.SKIP,
        reason="Condition is false.",
    )

    result = introspection_agent.pre_step_introspection(
        plan=mock_plan,
        plan_run=mock_plan_run,
    )

    assert result.outcome == PreStepIntrospectionOutcome.SKIP
    assert result.reason == "Condition is false."


def test_pre_step_introspection_stop(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
    mock_introspection_model: MagicMock,
) -> None:
    """Test pre_step_introspection returns STOP when remaining steps cannot be executed."""
    mock_introspection_model.get_structured_response.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.COMPLETE,
        reason="Remaining steps cannot be executed.",
    )

    result = introspection_agent.pre_step_introspection(
        plan=mock_plan,
        plan_run=mock_plan_run,
    )

    assert result.outcome == PreStepIntrospectionOutcome.COMPLETE
    assert result.reason == "Remaining steps cannot be executed."


def test_pre_step_introspection_passes_correct_data(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
    mock_introspection_model: MagicMock,
) -> None:
    """Test pre_step_introspection passes correct data to LLM."""
    mock_messages = [HumanMessage(content="Test message")]

    mock_introspection_model.get_structured_response.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.CONTINUE,
        reason="Test reason",
    )

    with patch(
        "langchain.prompts.ChatPromptTemplate.format_messages",
        return_value=mock_messages,
    ):
        result = introspection_agent.pre_step_introspection(
            plan=mock_plan,
            plan_run=mock_plan_run,
        )

        mock_introspection_model.get_structured_response.assert_called_once_with(
            schema=PreStepIntrospection,
            messages=[Message(role="user", content="Test message")],
        )

        assert result.outcome == PreStepIntrospectionOutcome.CONTINUE
        assert result.reason == "Test reason"


def test_retrieves_outputs_from_memory_correctly(
    introspection_agent: DefaultIntrospectionAgent,
    mock_introspection_model: MagicMock,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection returns CONTINUE when conditions are met."""
    stored_output = introspection_agent.agent_memory.save_plan_run_output(
        "$result1",
        mock_plan_run.outputs.step_outputs["$result1"],
        mock_plan_run.id,
    )
    mock_plan_run.outputs.step_outputs["$result1"] = stored_output

    mock_introspection_model.get_structured_response.return_value = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.CONTINUE,
        reason="All conditions are met.",
    )
    result = introspection_agent.pre_step_introspection(
        plan=mock_plan,
        plan_run=mock_plan_run,
    )

    assert result.outcome == PreStepIntrospectionOutcome.CONTINUE
    assert result.reason == "All conditions are met."
    assert (
        "Task 1 result"
        in mock_introspection_model.get_structured_response.call_args[1]["messages"][1].content
    )
