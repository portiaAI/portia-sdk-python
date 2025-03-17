"""Tests for the DefaultIntrospectionAgent module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from portia.execution_agents.base_execution_agent import Output
from portia.introspection_agents.default_introspection_agent import DefaultIntrospectionAgent
from portia.introspection_agents.introspection_agent import (
    BaseIntrospectionAgent,
    PreStepIntrospection,
    PreStepIntrospectionOutcome,
)
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan, PlanContext, Step, Variable
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState
from portia.prefixed_uuid import PlanUUID
from tests.utils import get_test_config

if TYPE_CHECKING:
    from portia.config import Config


@pytest.fixture
def mock_config() -> Config:
    """Mock Config object for testing."""
    return MagicMock()


@pytest.fixture
def introspection_agent(mock_config: Config) -> DefaultIntrospectionAgent:
    """Create an instance of the DefaultIntrospectionAgent with mocked config."""
    return DefaultIntrospectionAgent(config=mock_config)


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
        outputs=PlanRunOutputs(
            step_outputs={
                "$result1": Output(value="Task 1 result", summary="Task 1 summary"),
            },
            final_output=None,
        ),
    )


def test_base_introspection_agent() -> None:
    """Test BaseIntrospectionAgent raises."""

    class MyIntrospectionAgent(BaseIntrospectionAgent):
        """Override to test base."""

        def pre_step_introspection(
            self,
            plan: Plan,
            plan_run: PlanRun,
        ) -> PreStepIntrospection:
            return super().pre_step_introspection(plan, plan_run)

    agent = MyIntrospectionAgent(get_test_config())

    empty_plan = Plan(
        plan_context=PlanContext(query="test", tool_ids=[]),
        steps=[],
    )
    empty_plan_run = PlanRun(plan_id=empty_plan.id)

    with pytest.raises(NotImplementedError):
        agent.pre_step_introspection(empty_plan, empty_plan_run)


def test_pre_step_introspection_continue(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection returns CONTINUE when conditions are met."""
    # Mock the LLMWrapper response to simulate a CONTINUE outcome
    mock_response = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.CONTINUE,
        reason="All conditions are met.",
    )

    with patch.object(
        LLMWrapper, "to_langchain", return_value=MagicMock(
            with_structured_output=MagicMock(
                return_value=MagicMock(
                    invoke=MagicMock(return_value=mock_response),
                ),
            ),
        ),
    ):
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
) -> None:
    """Test pre_step_introspection returns SKIP when condition is false."""
    # Mock the LLMWrapper response to simulate a SKIP outcome
    mock_response = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.SKIP,
        reason="Condition is false.",
    )

    with patch.object(
        LLMWrapper, "to_langchain", return_value=MagicMock(
            with_structured_output=MagicMock(
                return_value=MagicMock(
                    invoke=MagicMock(return_value=mock_response),
                ),
            ),
        ),
    ):
        result = introspection_agent.pre_step_introspection(
            plan=mock_plan,
            plan_run=mock_plan_run,
        )

        assert result.outcome == PreStepIntrospectionOutcome.SKIP
        assert result.reason == "Condition is false."


def test_pre_step_introspection_fail(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection returns FAIL when missing required data."""
    # Mock the LLMWrapper response to simulate a FAIL outcome
    mock_response = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.FAIL,
        reason="Missing required data.",
    )

    with patch.object(
        LLMWrapper, "to_langchain", return_value=MagicMock(
            with_structured_output=MagicMock(
                return_value=MagicMock(
                    invoke=MagicMock(return_value=mock_response),
                ),
            ),
        ),
    ):
        result = introspection_agent.pre_step_introspection(
            plan=mock_plan,
            plan_run=mock_plan_run,
        )

        assert result.outcome == PreStepIntrospectionOutcome.FAIL
        assert result.reason == "Missing required data."


def test_pre_step_introspection_stop(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection returns STOP when remaining steps cannot be executed."""
    # Mock the LLMWrapper response to simulate a STOP outcome
    mock_response = PreStepIntrospection(
        outcome=PreStepIntrospectionOutcome.STOP,
        reason="Remaining steps cannot be executed.",
    )

    with patch.object(
        LLMWrapper, "to_langchain", return_value=MagicMock(
            with_structured_output=MagicMock(
                return_value=MagicMock(
                    invoke=MagicMock(return_value=mock_response),
                ),
            ),
        ),
    ):
        result = introspection_agent.pre_step_introspection(
            plan=mock_plan,
            plan_run=mock_plan_run,
        )

        assert result.outcome == PreStepIntrospectionOutcome.STOP
        assert result.reason == "Remaining steps cannot be executed."


def test_pre_step_introspection_passes_correct_data(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection passes correct data to LLM."""
    mock_messages = [MagicMock()]

    llm_mock = MagicMock()
    llm_mock.to_langchain.return_value.with_structured_output.return_value.invoke.return_value = (
        PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.CONTINUE,
            reason="Test reason",
        )
    )

    with patch.object(LLMWrapper, "for_usage", return_value=llm_mock), patch(
        "langchain.prompts.ChatPromptTemplate.format_messages", return_value=mock_messages,
    ):
        result = introspection_agent.pre_step_introspection(
            plan=mock_plan,
            plan_run=mock_plan_run,
        )

        llm_mock.to_langchain.return_value.with_structured_output.assert_called_once_with(
            PreStepIntrospection,
        )

        llm_mock.to_langchain.return_value.with_structured_output.return_value.invoke.assert_called_once_with(
            mock_messages,
        )

        assert result.outcome == PreStepIntrospectionOutcome.CONTINUE
        assert result.reason == "Test reason"


def test_pre_step_introspection_model_validate(
    introspection_agent: DefaultIntrospectionAgent,
    mock_plan: Plan,
    mock_plan_run: PlanRun,
) -> None:
    """Test pre_step_introspection validates the LLM response with model_validate."""
    raw_response = {
        "outcome": "CONTINUE",
        "reason": "All conditions are met.",
    }

    with patch.object(
        LLMWrapper, "to_langchain", return_value=MagicMock(
            with_structured_output=MagicMock(
                return_value=MagicMock(
                    invoke=MagicMock(return_value=raw_response),
                ),
            ),
        ),
    ), patch(
        "portia.introspection_agents.introspection_agent.PreStepIntrospection.model_validate",
        return_value=PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.CONTINUE,
            reason="All conditions are met.",
        ),
    ) as model_validate_mock:
        result = introspection_agent.pre_step_introspection(
            plan=mock_plan,
            plan_run=mock_plan_run,
        )

        # Verify model_validate was called with the raw response
        model_validate_mock.assert_called_once_with(raw_response)

        # Verify the result is the validated response
        assert result.outcome == PreStepIntrospectionOutcome.CONTINUE
        assert result.reason == "All conditions are met."
