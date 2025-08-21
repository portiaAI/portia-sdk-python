"""Integration tests for the ConditionalEvaluationAgent."""

import pytest

from portia import LLMProvider
from portia.config import Config
from portia.execution_agents.conditional_evaluation_agent import ConditionalEvaluationAgent
from tests.integration.test_e2e import PROVIDER_MODELS


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PROVIDER_MODELS)
@pytest.mark.asyncio
async def test_conditional_evaluation_agent(
    llm_provider: LLMProvider, default_model_name: str
) -> None:
    """The agent should correctly evaluate true and false statements."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
    )
    agent = ConditionalEvaluationAgent(config)

    true_result = await agent.execute("2 + 2 == 4", {})
    false_result = await agent.execute("2 + 2 == 5", {})

    assert true_result is True
    assert false_result is False


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PROVIDER_MODELS)
@pytest.mark.asyncio
async def test_conditional_evaluation_agent_with_arguments(
    llm_provider: LLMProvider, default_model_name: str
) -> None:
    """The agent should correctly evaluate statements using passed arguments."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
    )
    agent = ConditionalEvaluationAgent(config)

    true_result = await agent.execute(
        "x + y == z",
        {"x": 2, "y": 2, "z": 4},
    )
    false_result = await agent.execute(
        "x + y == z",
        {"x": 2, "y": 2, "z": 5},
    )

    assert true_result is True
    assert false_result is False
