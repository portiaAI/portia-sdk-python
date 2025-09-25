"""Unit tests for the cost estimator module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from portia.builder.llm_step import LLMStep
from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.react_agent_step import ReActAgentStep
from portia.config import Config
from portia.cost_estimator import (
    CostEstimator,
    PlanCostEstimate,
    StepCostEstimate,
)
from portia.plan import Plan, PlanContext, PlanInput, Step


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


def test_init() -> None:
    """Test CostEstimator initialization."""
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    estimator = CostEstimator()
    assert estimator.config is not None
    assert estimator.estimation_tool is not None

    custom_config = Config.from_default(default_model="openai/gpt-4o")
    estimator_with_config = CostEstimator(custom_config)
    assert estimator_with_config.config == custom_config


def test_calculate_cost(estimator: CostEstimator) -> None:
    """Test basic cost calculation."""
    pricing = estimator._get_model_pricing("gpt-4o")
    cost = estimator._calculate_cost(1000, 500, "gpt-4o")
    expected = (1000 / 1_000_000) * pricing["input"] + (500 / 1_000_000) * pricing["output"]
    assert cost == expected

    cost_unknown = estimator._calculate_cost(1000, 500, "unknown-model")
    expected_unknown = (1000 / 1_000_000) * 5.00 + (500 / 1_000_000) * 15.00
    assert cost_unknown == expected_unknown


def test_calculate_introspection_cost(estimator: CostEstimator) -> None:
    """Test introspection cost calculation."""
    pricing = estimator._get_model_pricing("gpt-4o")
    cost = estimator._calculate_introspection_cost("gpt-4o")
    expected = (800 / 1_000_000) * pricing["input"] + (100 / 1_000_000) * pricing["output"]
    assert cost == expected


def test_estimate_v2_step_with_condition_coverage(estimator: CostEstimator) -> None:
    """Test V2 step estimation with condition to cover introspection cost line 263."""
    from portia.builder.conditionals import ConditionalBlock
    from portia.builder.llm_step import LLMStep

    step = LLMStep(step_name="test_step", task="Test task")
    step.conditional_block = ConditionalBlock(clause_step_indexes=[0, 1])

    with patch.object(estimator, "_get_llm_estimation") as mock_estimation:
        mock_estimation.return_value = {
            "estimated_input_tokens": 1000,
            "estimated_output_tokens": 300,
            "number_of_llm_calls": 1,
            "reasoning": "Test reasoning",
        }

        result = estimator._estimate_v2_step_cost(step, "gpt-4o")

        assert result.has_condition is True
        assert result.introspection_cost > 0


def test_extract_step_task(estimator: CostEstimator) -> None:
    """Test task extraction from different step types."""
    llm_step = LLMStep(step_name="test", task="Test task")
    assert estimator._extract_step_task(llm_step) == "Test task"

    react_step = ReActAgentStep(step_name="test", task="React task", tools=[])
    assert estimator._extract_step_task(react_step) == "React task"

    mock_step = MagicMock()
    mock_step.__class__.__name__ = "MockStep"
    type(mock_step).task = property(
        lambda _: (_ for _ in ()).throw(
            AttributeError("'MockStep' object has no attribute 'task'")
        )
    )
    result = estimator._extract_step_task(mock_step)
    assert result == "Execute MockStep"


def test_extract_step_task_coverage(estimator: CostEstimator) -> None:
    """Test additional coverage of step task extraction."""
    mock_step = MagicMock()
    mock_step.__class__.__name__ = "CustomStep"
    type(mock_step).task = property(
        lambda _: (_ for _ in ()).throw(
            AttributeError("'CustomStep' object has no attribute 'task'")
        )
    )

    result = estimator._extract_step_task(mock_step)
    assert result == "Execute CustomStep"


def test_estimate_input_context(estimator: CostEstimator) -> None:
    """Test input context estimation."""
    step_no_inputs = LLMStep(step_name="test", task="Test")
    context = estimator._estimate_input_context(step_no_inputs)
    assert context == 1000

    step_with_inputs = LLMStep(step_name="test", task="Test", inputs=["input1", "input2"])
    context_with_inputs = estimator._estimate_input_context(step_with_inputs)
    assert context_with_inputs == 1400


def test_get_fallback_estimation(estimator: CostEstimator) -> None:
    """Test fallback estimation for different step types."""
    llm_estimation = estimator._get_fallback_estimation("LLMStep")
    assert llm_estimation["number_of_llm_calls"] == 1
    assert llm_estimation["estimated_input_tokens"] == 1000

    react_estimation = estimator._get_fallback_estimation("ReActAgentStep")
    assert react_estimation["number_of_llm_calls"] == 4
    assert react_estimation["estimated_input_tokens"] == 2000

    unknown_estimation = estimator._get_fallback_estimation("UnknownStep")
    assert unknown_estimation["number_of_llm_calls"] == 1
    assert "Unknown step type" in unknown_estimation["reasoning"]


@patch("portia.cost_estimator.LLMTool.run")
def test_get_llm_estimation_success(mock_llm_run: MagicMock, estimator: CostEstimator) -> None:
    """Test successful LLM-based estimation."""
    from portia.cost_estimator import LLMEstimationResult

    mock_response = LLMEstimationResult(
        estimated_input_tokens=1500,
        estimated_output_tokens=400,
        number_of_llm_calls=3,
        reasoning="Test reasoning",
    )
    mock_llm_run.return_value = mock_response

    result = estimator._get_llm_estimation("TestStep", "Test task", "gpt-4o", "Test tools", 1000)

    assert result["estimated_input_tokens"] == 1500
    assert result["estimated_output_tokens"] == 400
    assert result["number_of_llm_calls"] == 3
    assert result["reasoning"] == "Test reasoning"


@patch("portia.cost_estimator.LLMTool.run")
def test_get_llm_estimation_non_structured_fallback(
    mock_llm_run: MagicMock, estimator: CostEstimator
) -> None:
    """Test fallback when LLM returns non-structured output."""
    mock_response = "This is not a structured response"
    mock_llm_run.return_value = mock_response

    result = estimator._get_llm_estimation("LLMStep", "Test task", "gpt-4o", "Test tools", 1000)

    assert result["estimated_input_tokens"] == 1000
    assert result["estimated_output_tokens"] == 300
    assert result["number_of_llm_calls"] == 1


@patch("portia.cost_estimator.LLMTool.run")
def test_get_llm_estimation_failure(mock_llm_run: MagicMock, estimator: CostEstimator) -> None:
    """Test LLM estimation failure fallback."""
    mock_llm_run.side_effect = ValueError("LLM failed")

    result = estimator._get_llm_estimation("LLMStep", "Test task", "gpt-4o", "Test tools", 1000)

    assert result["estimated_input_tokens"] == 1000
    assert result["number_of_llm_calls"] == 1


def test_get_fallback_estimation_all_types(estimator: CostEstimator) -> None:
    """Test fallback estimation for all step types."""
    llm_result = estimator._get_fallback_estimation("LLMStep")
    assert llm_result["estimated_input_tokens"] == 1000
    assert llm_result["estimated_output_tokens"] == 300
    assert llm_result["number_of_llm_calls"] == 1

    react_result = estimator._get_fallback_estimation("ReActAgentStep")
    assert react_result["estimated_input_tokens"] == 2000
    assert react_result["estimated_output_tokens"] == 800
    assert react_result["number_of_llm_calls"] == 4

    tool_result = estimator._get_fallback_estimation("SingleToolAgentStep")
    assert tool_result["estimated_input_tokens"] == 1500
    assert tool_result["estimated_output_tokens"] == 400
    assert tool_result["number_of_llm_calls"] == 1

    execution_result = estimator._get_fallback_estimation("ExecutionAgentStep")
    assert execution_result["estimated_input_tokens"] == 1200
    assert execution_result["estimated_output_tokens"] == 350
    assert execution_result["number_of_llm_calls"] == 1

    unknown_result = estimator._get_fallback_estimation("UnknownStep")
    assert unknown_result["estimated_input_tokens"] == 1000
    assert unknown_result["estimated_output_tokens"] == 300
    assert unknown_result["number_of_llm_calls"] == 1


def test_estimate_input_context_additional_coverage(estimator: CostEstimator) -> None:
    """Test additional coverage for input context estimation."""
    step_many_inputs = LLMStep(
        step_name="test",
        task="Test",
        inputs=[f"input{i}" for i in range(10)],
    )
    context_many_inputs = estimator._estimate_input_context(step_many_inputs)
    expected = 1000 + (10 * 200)  # Base + (number_of_inputs * 200)
    assert context_many_inputs == expected


@patch("portia.cost_estimator.LLMTool.run")
def test_llm_estimation_malformed_json_fallback(
    mock_llm_run: MagicMock, estimator: CostEstimator
) -> None:
    """Test LLM estimation with malformed JSON response triggers fallback."""
    mock_llm_run.return_value = "This is not JSON at all"

    result = estimator._get_llm_estimation("LLMStep", "Test task", "gpt-4o", "Test tools", 1000)

    assert result["estimated_input_tokens"] == 1000
    assert result["estimated_output_tokens"] == 300
    assert result["number_of_llm_calls"] == 1


def test_estimate_v1_step_cost(estimator: CostEstimator) -> None:
    """Test V1 step cost estimation."""
    mock_step = MagicMock()
    mock_step.output = "test_output"
    mock_step.task = "Test task"
    mock_step.condition = None

    estimate = estimator._estimate_v1_step_cost(mock_step, "gpt-4o")

    assert isinstance(estimate, StepCostEstimate)
    assert estimate.step_name == "Step test_output"
    assert estimate.step_type == "V1Step"
    assert estimate.estimated_cost > 0
    assert not estimate.has_condition
    assert estimate.introspection_cost == 0.0


def test_estimate_v1_step_cost_with_condition(estimator: CostEstimator) -> None:
    """Test V1 step cost estimation with condition."""
    mock_step = MagicMock()
    mock_step.output = "test_output"
    mock_step.task = "Test task"
    mock_step.condition = "some condition"

    estimate = estimator._estimate_v1_step_cost(mock_step, "gpt-4o")

    assert estimate.has_condition
    assert estimate.introspection_cost > 0


@patch.object(CostEstimator, "_get_llm_estimation")
def test_estimate_v2_step_cost(mock_llm_estimation: MagicMock, estimator: CostEstimator) -> None:
    """Test V2 step cost estimation."""
    mock_llm_estimation.return_value = {
        "estimated_input_tokens": 1000,
        "estimated_output_tokens": 300,
        "number_of_llm_calls": 2,
        "reasoning": "Test reasoning",
    }

    step = LLMStep(step_name="test_step", task="Test task")
    estimate = estimator._estimate_v2_step_cost(step, "gpt-4o")

    assert isinstance(estimate, StepCostEstimate)
    assert estimate.step_name == "test_step"
    assert estimate.step_type == "LLMStep"
    assert estimate.estimated_input_tokens == 2000
    assert estimate.estimated_output_tokens == 600
    assert estimate.estimated_cost > 0
    assert "Test reasoning" in estimate.explanation


def test_estimate_v1_plan(estimator: CostEstimator) -> None:
    """Test V1 plan cost estimation."""
    plan = Plan(
        plan_context=PlanContext(query="Test query", tool_ids=[]),
        plan_inputs=[PlanInput(name="input1", description="Test input")],
        steps=[
            Step(task="Step 1", output="$output1"),
            Step(task="Step 2", output="$output2", condition="some condition"),
        ],
    )

    estimate = estimator._estimate_v1_plan(plan)

    assert isinstance(estimate, PlanCostEstimate)
    assert len(estimate.step_estimates) == 2
    assert estimate.total_estimated_cost > 0
    assert estimate.step_estimates[1].has_condition
    assert estimate.methodology
    assert estimate.limitations


def test_estimate_v2_plan(estimator: CostEstimator) -> None:
    """Test V2 plan cost estimation."""
    plan = (
        PlanBuilderV2("Test plan")
        .llm_step(step_name="step1", task="First task")
        .react_agent_step(step_name="step2", task="Second task", tools=["tool1"])
        .build()
    )

    with patch.object(estimator, "_get_llm_estimation") as mock_estimation:
        mock_estimation.return_value = {
            "estimated_input_tokens": 1000,
            "estimated_output_tokens": 300,
            "number_of_llm_calls": 2,
            "reasoning": "Test reasoning",
        }

        estimate = estimator._estimate_v2_plan(plan)

    assert isinstance(estimate, PlanCostEstimate)
    assert len(estimate.step_estimates) == 2
    assert estimate.total_estimated_cost > 0
    assert estimate.step_estimates[0].step_type == "LLMStep"
    assert estimate.step_estimates[1].step_type == "ReActAgentStep"


def test_plan_estimate_v1(estimator: CostEstimator) -> None:
    """Test main plan_estimate method with V1 plan."""
    plan = Plan(
        plan_context=PlanContext(query="Test query", tool_ids=[]),
        steps=[Step(task="Test step", output="$output")],
    )

    estimate = estimator.plan_estimate(plan)
    assert isinstance(estimate, PlanCostEstimate)
    assert len(estimate.step_estimates) == 1


def test_plan_estimate_v1_coverage_path(estimator: CostEstimator) -> None:
    """Ensure V1 plan estimation path is fully covered."""
    from portia.plan import Plan

    plan = Plan(
        plan_context=PlanContext(query="V1 coverage test", tool_ids=[]),
        plan_inputs=[PlanInput(name="test_input", description="Coverage input")],
        steps=[
            Step(task="Coverage task", inputs=[], output="coverage_output"),
        ],
    )

    estimate = estimator.plan_estimate(plan)
    assert isinstance(estimate, PlanCostEstimate)
    assert estimate.total_estimated_cost > 0


def test_plan_estimate_v2(estimator: CostEstimator) -> None:
    """Test main plan_estimate method with V2 plan."""
    plan = PlanBuilderV2("Test plan").llm_step(step_name="step1", task="Test task").build()

    with patch.object(estimator, "_get_llm_estimation") as mock_estimation:
        mock_estimation.return_value = {
            "estimated_input_tokens": 1000,
            "estimated_output_tokens": 300,
            "number_of_llm_calls": 1,
            "reasoning": "Test reasoning",
        }

        estimate = estimator.plan_estimate(plan)

    assert isinstance(estimate, PlanCostEstimate)
    assert len(estimate.step_estimates) == 1


@pytest.mark.asyncio
async def test_aplan_estimate(estimator: CostEstimator) -> None:
    """Test async plan estimation."""
    plan = Plan(
        plan_context=PlanContext(query="Test query", tool_ids=[]),
        steps=[Step(task="Test step", output="$output")],
    )

    estimate = await estimator.aplan_estimate(plan)
    assert isinstance(estimate, PlanCostEstimate)


def test_methodology_and_limitations(estimator: CostEstimator) -> None:
    """Test methodology and limitations explanations."""
    methodology = estimator._get_methodology_explanation()
    assert "Cost estimation methodology" in methodology
    assert "LLM-driven estimation" in methodology

    limitations = estimator._get_limitations_explanation()
    assert "Limitations and assumptions" in limitations
    assert "Only includes LLM costs" in limitations


def test_v1_plan_path_explicit_ci_coverage(estimator: CostEstimator) -> None:
    """Explicitly test V1 plan path to ensure CI coverage of line 148."""
    from portia.plan import Plan, PlanContext, PlanInput, Step

    v1_plan = Plan(
        plan_context=PlanContext(query="V1 explicit test", tool_ids=[]),
        plan_inputs=[PlanInput(name="test", description="Test input")],
        steps=[Step(task="V1 test task", inputs=[], output="v1_output")],
    )

    estimate = estimator.plan_estimate(v1_plan)

    assert isinstance(estimate, PlanCostEstimate)
    assert len(estimate.step_estimates) == 1
    assert "v1_output" in estimate.step_estimates[0].step_name


def test_step_cost_estimate_creation() -> None:
    """Test creating a StepCostEstimate."""
    estimate = StepCostEstimate(
        step_name="test_step",
        step_type="LLMStep",
        estimated_input_tokens=1000,
        estimated_output_tokens=300,
        estimated_cost=0.025,
        cost_breakdown={"execution": 0.025, "introspection": 0.0},
        explanation="Test explanation",
        has_condition=False,
    )

    assert estimate.step_name == "test_step"
    assert estimate.step_type == "LLMStep"
    assert estimate.estimated_cost == 0.025
    assert not estimate.has_condition


def test_plan_cost_estimate_creation() -> None:
    """Test creating a PlanCostEstimate."""
    step_estimate = StepCostEstimate(
        step_name="test_step",
        step_type="LLMStep",
        estimated_input_tokens=1000,
        estimated_output_tokens=300,
        estimated_cost=0.025,
        cost_breakdown={"execution": 0.025},
        explanation="Test explanation",
        has_condition=False,
    )

    plan_estimate = PlanCostEstimate(
        total_estimated_cost=0.025,
        step_estimates=[step_estimate],
        model_used="gpt-4o",
        methodology="Test methodology",
        limitations="Test limitations",
    )

    assert plan_estimate.total_estimated_cost == 0.025
    assert len(plan_estimate.step_estimates) == 1
    assert plan_estimate.model_used == "gpt-4o"


def test_get_model_pricing_litellm_integration() -> None:
    """Test that _get_model_pricing works with LiteLLM integration."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    estimator = CostEstimator()

    pricing = estimator._get_model_pricing("gpt-4o")
    assert isinstance(pricing, dict)
    assert "input" in pricing
    assert "output" in pricing
    assert isinstance(pricing["input"], int | float)
    assert isinstance(pricing["output"], int | float)
    assert pricing["input"] > 0
    assert pricing["output"] > 0


def test_get_model_pricing_fallback() -> None:
    """Test that pricing fallback works for unknown models."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    estimator = CostEstimator()
    pricing = estimator._get_model_pricing("unknown-model-xyz")
    assert pricing == {"input": 5.00, "output": 15.00}


def test_pricing_reasonableness() -> None:
    """Test that pricing values are reasonable."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    estimator = CostEstimator()

    test_models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-latest"]

    for model in test_models:
        pricing = estimator._get_model_pricing(model)
        assert pricing["output"] >= pricing["input"]
        assert 0 < pricing["input"] < 100
        assert 0 < pricing["output"] < 200


def test_additional_step_types_coverage(estimator: CostEstimator) -> None:
    """Test coverage for different step types and edge cases."""
    from portia.builder.llm_step import LLMStep
    from portia.builder.react_agent_step import ReActAgentStep
    from portia.builder.conditionals import ConditionalBlock
    
    step_with_conditional = LLMStep(step_name="conditional_step", task="Test conditional task")
    step_with_conditional.conditional_block = ConditionalBlock(clause_step_indexes=[0, 1])
    
    estimate = estimator._estimate_v2_step_cost(step_with_conditional, "gpt-4o")
    assert estimate.has_condition
    assert estimate.introspection_cost >= 0 
    
    react_step = ReActAgentStep(step_name="react_step", task="Test ReAct task", tools=["search_tool"])
    estimate = estimator._estimate_v2_step_cost(react_step, "gpt-4o")
    assert estimate.step_type == "ReActAgentStep"
    assert estimate.estimated_cost > 0


def test_llm_estimation_fallback_paths(estimator: CostEstimator) -> None:
    """Test various fallback scenarios in LLM estimation."""
    from unittest.mock import patch
    
    with patch('portia.open_source_tools.llm_tool.LLMTool.run') as mock_run:
        mock_run.return_value = "Just a string response, not structured"
        
        result = estimator._get_llm_estimation("LLMStep", "test task", "gpt-4o", "test_tool", 1000)
        
        assert "estimated_input_tokens" in result
        assert "estimated_output_tokens" in result
        assert result["number_of_llm_calls"] == 1
    
    with patch('portia.open_source_tools.llm_tool.LLMTool.run') as mock_run:
        mock_run.side_effect = ValueError("API error")
        
        result = estimator._get_llm_estimation("LLMStep", "test task", "gpt-4o", "test_tool", 1000)
        
        assert "estimated_input_tokens" in result
        assert "estimated_output_tokens" in result
        assert result["number_of_llm_calls"] == 1


def test_model_pricing_exception_handling(estimator: CostEstimator) -> None:
    """Test exception handling in model pricing lookup."""
    from unittest.mock import patch
    
    with patch('litellm.get_model_cost_map') as mock_get_cost:
       
        mock_get_cost.side_effect = AttributeError("LiteLLM error")
        pricing = estimator._get_model_pricing("some-model")
        
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0


def test_step_conversion_failure_coverage(estimator: CostEstimator) -> None:
    """Test coverage for step conversion failure scenario."""
    from unittest.mock import Mock, patch
    from portia.builder.llm_step import LLMStep
    
    mock_step = Mock(spec=LLMStep)
    mock_step.step_name = "failing_step"
    mock_step.__class__.__name__ = "LLMStep"
    mock_step.task = "Test task"
    
    mock_step.to_legacy_step = Mock(side_effect=ValueError("Conversion failed"))
    
    result = estimator._estimate_v2_step_cost(mock_step, "gpt-4o")
    
    assert result.estimated_cost > 0
    assert result.step_name == "failing_step"


def test_edge_case_coverage_final() -> None:
    """Final test to push coverage to 100% - test edge cases."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        import pytest
        pytest.skip("OPENAI_API_KEY not set")
    
    from portia import Config
    from portia.builder.llm_step import LLMStep  
    from portia.builder.conditionals import ConditionalBlock
    from portia.plan import Step
    
    config = Config.from_default(default_model="openai/gpt-4o")
    estimator = CostEstimator(config)
    
    v1_step = Step(task="Test task", output="$result", condition="len($result) > 0")
    estimate = estimator._estimate_v1_step_cost(v1_step, "gpt-4o")
    assert estimate.has_condition
    
    v2_step = LLMStep(step_name="test", task="Test task")
    v2_step.conditional_block = ConditionalBlock(clause_step_indexes=[0])
    estimate = estimator._estimate_v2_step_cost(v2_step, "gpt-4o")
    assert estimate.has_condition
