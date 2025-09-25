"""Integration tests for the cost estimator with real plans."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from portia import (
    Config,
    CostEstimator,
    Input,
    PlanBuilder,
    PlanBuilderV2,
)


class CommodityPrice(BaseModel):
    """Test output schema."""

    price: float
    currency: str


@pytest.fixture
def config() -> Config:
    """Set up test configuration."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    return Config.from_default(default_model="openai/gpt-4o")


@pytest.fixture
def estimator(config: Config) -> CostEstimator:
    """Set up cost estimator with test configuration."""
    return CostEstimator(config)


def test_simple_llm_plan(estimator: CostEstimator) -> None:
    """Test cost estimation for a simple LLM plan."""
    plan = (
        PlanBuilderV2("Simple LLM task")
        .llm_step(
            step_name="analyze_text",
            task="Analyze the sentiment of this text: 'I love sunny days!'",
        )
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0
    assert len(estimate.step_estimates) == 1
    assert estimate.step_estimates[0].step_type == "LLMStep"
    assert estimate.step_estimates[0].step_name == "analyze_text"
    assert estimate.methodology
    assert estimate.limitations


def test_complex_multi_step_plan(estimator: CostEstimator) -> None:
    """Test cost estimation for a complex multi-step plan."""
    plan = (
        PlanBuilderV2("Complex workflow")
        .input(name="topic", description="Topic to research", default_value="AI")
        .llm_step(
            step_name="generate_questions",
            task=f"Generate 3 research questions about {Input('topic')}",
        )
        .react_agent_step(
            step_name="research_topic",
            task="Research the topic using available tools",
            tools=["search_tool", "calculator_tool"],
            inputs=[Input("topic")],
            output_schema=CommodityPrice,
        )
        .single_tool_agent_step(
            step_name="summarize_findings",
            task="Create a summary of the research findings",
            tool="llm_tool",
        )
        .user_input(
            step_name="get_feedback",
            message="Please provide feedback on the research",
        )
        .llm_step(
            step_name="final_report",
            task="Create a final report incorporating user feedback",
        )
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0
    assert len(estimate.step_estimates) == 5

    step_types = [step.step_type for step in estimate.step_estimates]
    assert "LLMStep" in step_types
    assert "ReActAgentStep" in step_types
    assert "SingleToolAgentStep" in step_types
    assert "UserInputStep" in step_types

    react_step = next(
        step for step in estimate.step_estimates if step.step_type == "ReActAgentStep"
    )
    llm_steps = [step for step in estimate.step_estimates if step.step_type == "LLMStep"]

    assert react_step.estimated_cost > min(step.estimated_cost for step in llm_steps)


def test_conditional_plan(estimator: CostEstimator) -> None:
    """Test cost estimation for a plan with conditional logic."""
    plan = (
        PlanBuilderV2("Conditional workflow")
        .function_step(
            step_name="get_number",
            function=lambda: 42,
        )
        .if_(
            condition=lambda x: x > 50,
            args={"x": Input("get_number")},
        )
        .llm_step(
            step_name="high_number_task",
            task="Process a high number",
        )
        .else_()
        .llm_step(
            step_name="low_number_task",
            task="Process a low number",
        )
        .endif()
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0

    conditional_steps = [
        step
        for step in estimate.step_estimates
        if step.has_condition or step.introspection_cost > 0
    ]

    assert len(conditional_steps) > 0


def test_loop_plan(estimator: CostEstimator) -> None:
    """Test cost estimation for a plan with loops."""
    plan = (
        PlanBuilderV2("Loop workflow")
        .function_step(
            step_name="create_items",
            function=lambda: [1, 2, 3],
        )
        .loop(over=Input("create_items"), step_name="process_loop")
        .llm_step(
            step_name="process_item",
            task="Process this item",
        )
        .end_loop(step_name="end_processing")
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0
    assert len(estimate.step_estimates) > 0


@pytest.mark.asyncio
async def test_async_estimation(estimator: CostEstimator) -> None:
    """Test async cost estimation."""
    plan = (
        PlanBuilderV2("Async test")
        .llm_step(
            step_name="async_task",
            task="This is an async test task",
        )
        .build()
    )

    estimate = await estimator.aplan_estimate(plan)

    assert estimate.total_estimated_cost > 0
    assert len(estimate.step_estimates) == 1


def test_estimation_consistency(estimator: CostEstimator) -> None:
    """Test that multiple estimations of the same plan are consistent."""
    plan = (
        PlanBuilderV2("Consistency test")
        .llm_step(
            step_name="consistent_task",
            task="This task should have consistent cost estimates",
        )
        .build()
    )

    estimate1 = estimator.plan_estimate(plan)
    estimate2 = estimator.plan_estimate(plan)

    cost_diff = abs(estimate1.total_estimated_cost - estimate2.total_estimated_cost)
    avg_cost = (estimate1.total_estimated_cost + estimate2.total_estimated_cost) / 2

    assert cost_diff / avg_cost < 0.5


def test_real_world_plan_estimation(estimator: CostEstimator) -> None:
    """Test cost estimation on a realistic plan similar to example_builder.py."""
    plan = (
        PlanBuilderV2("Buy some gold")
        .input(name="country", description="The country to purchase gold in", default_value="UK")
        .invoke_tool_step(
            step_name="search_currency",
            tool="search_tool",
            args={"search_query": f"What is the currency in {Input('country')}?"},
        )
        .react_agent_step(
            step_name="search_gold_price",
            tools=["search_tool", "calculator_tool"],
            task=f"What is the price of gold per ounce in {Input('country')}?",
            output_schema=CommodityPrice,
        )
        .user_input(
            step_name="purchase_quantity",
            message="How many ounces of gold do you want to purchase?",
            options=[50, 100, 200],
        )
        .function_step(
            step_name="calculate_total",
            function=lambda price_info, quantity: price_info.price * quantity,
            args={
                "price_info": Input("search_gold_price"),
                "quantity": Input("purchase_quantity"),
            },
        )
        .user_verify(message=f"Proceed with purchase? Total: {Input('calculate_total')}")
        .if_(
            condition=lambda total: total > 100,
            args={"total": Input("calculate_total")},
        )
        .function_step(function=lambda: "Big spender!")
        .else_()
        .function_step(function=lambda: "Small purchase")
        .endif()
        .llm_step(
            step_name="generate_receipt",
            task="Create a receipt for the gold purchase",
        )
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0
    assert len(estimate.step_estimates) > 5

    step_types = {step.step_type for step in estimate.step_estimates}
    assert "ReActAgentStep" in step_types
    assert "LLMStep" in step_types

    conditional_steps = [step for step in estimate.step_estimates if step.has_condition]
    assert len(conditional_steps) > 0

    assert 0.001 < estimate.total_estimated_cost < 1.0


def test_different_models(estimator: CostEstimator) -> None:
    """Test cost estimation with different models."""
    plan = (
        PlanBuilderV2("Model comparison")
        .llm_step(
            step_name="test_task",
            task="This is a test task",
            model="gpt-4o-mini",
        )
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0


def test_empty_plan(estimator: CostEstimator) -> None:
    """Test cost estimation for an empty plan."""
    plan = PlanBuilderV2("Empty plan").build()

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost == 0.0
    assert len(estimate.step_estimates) == 0
    assert estimate.model_used
    assert estimate.methodology
    assert estimate.limitations


def test_cost_breakdown_details(estimator: CostEstimator) -> None:
    """Test that cost breakdowns provide useful detail."""
    plan = (
        PlanBuilderV2("Breakdown test")
        .llm_step(
            step_name="normal_step",
            task="A normal LLM step",
        )
        .if_(condition=lambda: True)
        .llm_step(
            step_name="conditional_step",
            task="A conditional LLM step",
        )
        .endif()
        .build()
    )

    estimate = estimator.plan_estimate(plan)

    for step_estimate in estimate.step_estimates:
        assert "execution" in step_estimate.cost_breakdown
        assert step_estimate.explanation

        if step_estimate.has_condition:
            assert step_estimate.introspection_cost > 0
            if "introspection" in step_estimate.cost_breakdown:
                assert step_estimate.cost_breakdown["introspection"] > 0


def test_v1_plan_estimation(estimator: CostEstimator) -> None:
    """Test cost estimation for V1 legacy plans to improve coverage."""
    with pytest.warns(DeprecationWarning, match="Use PlanBuilderV2 instead"):
        plan = (
            PlanBuilder("Legacy V1 plan for coverage testing")
            .step(task="Analyze market trends", output="$analysis")
            .step(
                task="Generate investment recommendations based on $analysis",
                output="$recommendations",
                condition="len($analysis) > 100",
            )
            .build()
        )

    estimate = estimator.plan_estimate(plan)

    assert estimate.total_estimated_cost > 0
    assert len(estimate.step_estimates) == 2
    assert estimate.model_used == "gpt-4o"
    assert "V1Step" in [step.step_type for step in estimate.step_estimates]

    conditional_steps = [step for step in estimate.step_estimates if step.has_condition]
    assert len(conditional_steps) == 1
    assert conditional_steps[0].introspection_cost > 0

    async def test_async_v1() -> None:
        async_estimate = await estimator.aplan_estimate(plan)
        assert async_estimate.total_estimated_cost > 0
        assert len(async_estimate.step_estimates) == 2

    import asyncio

    asyncio.run(test_async_v1())
