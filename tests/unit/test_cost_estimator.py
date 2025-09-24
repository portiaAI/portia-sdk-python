"""Unit tests for the cost estimator module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from portia.builder.llm_step import LLMStep
from portia.builder.plan_builder_v2 import PlanBuilderV2
from portia.builder.react_agent_step import ReActAgentStep
from portia.builder.single_tool_agent_step import SingleToolAgentStep
from portia.config import Config
from portia.cost_estimator import (
    MODEL_PRICING,
    CostEstimator,
    PlanCostEstimate,
    StepCostEstimate,
)
from portia.plan import Plan, PlanContext, PlanInput, Step


class TestCostEstimator:
    """Test cases for the CostEstimator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        import os

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.config = Config.from_default(default_model="openai/gpt-4o")
        self.estimator = CostEstimator(self.config)

    def test_init(self) -> None:
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

    def test_calculate_cost(self) -> None:
        """Test basic cost calculation."""
        cost = self.estimator._calculate_cost(1000, 500, "gpt-4o")
        expected = (1000 / 1_000_000) * 5.00 + (500 / 1_000_000) * 15.00
        assert cost == expected

        cost_unknown = self.estimator._calculate_cost(1000, 500, "unknown-model")
        expected_unknown = (1000 / 1_000_000) * 5.00 + (500 / 1_000_000) * 15.00
        assert cost_unknown == expected_unknown

    def test_calculate_introspection_cost(self) -> None:
        """Test introspection cost calculation."""
        cost = self.estimator._calculate_introspection_cost("gpt-4o")
        expected = (800 / 1_000_000) * 5.00 + (100 / 1_000_000) * 15.00
        assert cost == expected

    def test_estimate_v2_step_with_condition_coverage(self) -> None:
        """Test V2 step estimation with condition to cover introspection cost line 263."""
        from portia.builder.llm_step import LLMStep
        from portia.builder.conditionals import ConditionalBlock

        step = LLMStep(step_name="test_step", task="Test task")
        step.conditional_block = ConditionalBlock(clause_step_indexes=[0, 1])

        with patch.object(self.estimator, "_get_llm_estimation") as mock_estimation:
            mock_estimation.return_value = {
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 300,
                "number_of_llm_calls": 1,
                "reasoning": "Test reasoning",
            }

            result = self.estimator._estimate_v2_step_cost(step, "gpt-4o")

            assert result.has_condition is True
            assert result.introspection_cost > 0

    def test_extract_step_task(self) -> None:
        """Test task extraction from different step types."""
        llm_step = LLMStep(step_name="test", task="Test task")
        assert self.estimator._extract_step_task(llm_step) == "Test task"

        react_step = ReActAgentStep(step_name="test", task="React task", tools=[])
        assert self.estimator._extract_step_task(react_step) == "React task"

        mock_step = MagicMock()
        mock_step.__class__.__name__ = "MockStep"
        del mock_step.task
        result = self.estimator._extract_step_task(mock_step)
        assert result == "Execute MockStep"

    def test_get_step_tools(self) -> None:
        """Test tool information extraction."""
        react_step = ReActAgentStep(step_name="test", task="Test", tools=["tool1", "tool2"])
        result = self.estimator._get_step_tools(react_step)
        assert result == "Tools: tool1, tool2"

        react_step_no_tools = ReActAgentStep(step_name="test", task="Test", tools=[])
        result_no_tools = self.estimator._get_step_tools(react_step_no_tools)
        assert result_no_tools == "Tools: None"

        single_tool_step = SingleToolAgentStep(step_name="test", task="Test", tool="my_tool")
        result_single = self.estimator._get_step_tools(single_tool_step)
        assert result_single == "Tool: my_tool"

        llm_step = LLMStep(step_name="test", task="Test")
        result_llm = self.estimator._get_step_tools(llm_step)
        assert result_llm == "No tools"

    def test_estimate_input_context(self) -> None:
        """Test input context estimation."""
        step_no_inputs = LLMStep(step_name="test", task="Test")
        context = self.estimator._estimate_input_context(step_no_inputs)
        assert context == 1000

        step_with_inputs = LLMStep(step_name="test", task="Test", inputs=["input1", "input2"])
        context_with_inputs = self.estimator._estimate_input_context(step_with_inputs)
        assert context_with_inputs == 1400

    def test_get_fallback_estimation(self) -> None:
        """Test fallback estimation for different step types."""
        llm_estimation = self.estimator._get_fallback_estimation("LLMStep")
        assert llm_estimation["number_of_llm_calls"] == 1
        assert llm_estimation["estimated_input_tokens"] == 1000

        react_estimation = self.estimator._get_fallback_estimation("ReActAgentStep")
        assert react_estimation["number_of_llm_calls"] == 4
        assert react_estimation["estimated_input_tokens"] == 2000

        unknown_estimation = self.estimator._get_fallback_estimation("UnknownStep")
        assert unknown_estimation["number_of_llm_calls"] == 2
        assert "Unknown step type" in unknown_estimation["reasoning"]

    @patch("portia.cost_estimator.LLMTool.run")
    def test_get_llm_estimation_success(self, mock_llm_run: MagicMock) -> None:
        """Test successful LLM-based estimation."""
        mock_response = json.dumps(
            {
                "estimated_input_tokens": 1500,
                "estimated_output_tokens": 400,
                "number_of_llm_calls": 3,
                "reasoning": "Test reasoning",
            }
        )
        mock_llm_run.return_value = mock_response

        result = self.estimator._get_llm_estimation(
            "TestStep", "Test task", "gpt-4o", "Test tools", 1000
        )

        assert result["estimated_input_tokens"] == 1500
        assert result["estimated_output_tokens"] == 400
        assert result["number_of_llm_calls"] == 3
        assert result["reasoning"] == "Test reasoning"

    @patch("portia.cost_estimator.LLMTool.run")
    def test_get_llm_estimation_json_wrapped(self, mock_llm_run: MagicMock) -> None:
        """Test LLM estimation with JSON wrapped in text."""
        mock_response = (
            'Here is the estimation: {"estimated_input_tokens": 1200, '
            '"estimated_output_tokens": 300, "number_of_llm_calls": 2, '
            '"reasoning": "Wrapped JSON"} Hope this helps!'
        )
        mock_llm_run.return_value = mock_response

        result = self.estimator._get_llm_estimation(
            "TestStep", "Test task", "gpt-4o", "Test tools", 1000
        )

        assert result["estimated_input_tokens"] == 1200
        assert result["estimated_output_tokens"] == 300

    @patch("portia.cost_estimator.LLMTool.run")
    def test_get_llm_estimation_failure(self, mock_llm_run: MagicMock) -> None:
        """Test LLM estimation failure fallback."""
        mock_llm_run.side_effect = Exception("LLM failed")

        result = self.estimator._get_llm_estimation(
            "LLMStep", "Test task", "gpt-4o", "Test tools", 1000
        )

        assert result["estimated_input_tokens"] == 1000
        assert result["number_of_llm_calls"] == 1

    def test_get_fallback_estimation_all_types(self) -> None:
        """Test fallback estimation for all step types."""
        llm_result = self.estimator._get_fallback_estimation("LLMStep")
        assert llm_result["estimated_input_tokens"] == 1000
        assert llm_result["estimated_output_tokens"] == 300
        assert llm_result["number_of_llm_calls"] == 1

        react_result = self.estimator._get_fallback_estimation("ReActAgentStep")
        assert react_result["estimated_input_tokens"] == 2000
        assert react_result["estimated_output_tokens"] == 800
        assert react_result["number_of_llm_calls"] == 4

        tool_result = self.estimator._get_fallback_estimation("SingleToolAgentStep")
        assert tool_result["estimated_input_tokens"] == 1500
        assert tool_result["estimated_output_tokens"] == 400
        assert tool_result["number_of_llm_calls"] == 2

        unknown_result = self.estimator._get_fallback_estimation("UnknownStep")
        assert unknown_result["estimated_input_tokens"] == 1000
        assert unknown_result["estimated_output_tokens"] == 300
        assert unknown_result["number_of_llm_calls"] == 2

    def test_get_step_tools_with_tool_objects(self) -> None:
        """Test _get_step_tools with actual Tool objects to cover tool.id access."""
        from portia.builder.react_agent_step import ReActAgentStep
        from portia.tool import Tool

        mock_tool1 = MagicMock(spec=Tool)
        mock_tool1.id = "tool1_id"
        mock_tool2 = MagicMock(spec=Tool)
        mock_tool2.id = "tool2_id"

        mock_step = MagicMock(spec=ReActAgentStep)
        mock_step.tools = [mock_tool1, "string_tool", mock_tool2]

        result = self.estimator._get_step_tools(mock_step)
        assert result == "Tools: tool1_id, string_tool, tool2_id"

    @patch("portia.cost_estimator.LLMTool.run")
    def test_llm_estimation_malformed_json_fallback(self, mock_llm_run: MagicMock) -> None:
        """Test LLM estimation with malformed JSON response triggers fallback."""
    
        mock_llm_run.return_value = "This is not JSON at all"

        result = self.estimator._get_llm_estimation(
            "LLMStep", "Test task", "gpt-4o", "Test tools", 1000
        )
       
        assert result["estimated_input_tokens"] == 1000  
        assert result["estimated_output_tokens"] == 300
        assert result["number_of_llm_calls"] == 1

    def test_estimate_v1_step_cost(self) -> None:
        """Test V1 step cost estimation."""
        mock_step = MagicMock()
        mock_step.output = "test_output"
        mock_step.task = "Test task"
        mock_step.condition = None

        estimate = self.estimator._estimate_v1_step_cost(mock_step, "gpt-4o")

        assert isinstance(estimate, StepCostEstimate)
        assert estimate.step_name == "Step test_output"
        assert estimate.step_type == "V1Step"
        assert estimate.estimated_cost > 0
        assert not estimate.has_condition
        assert estimate.introspection_cost == 0.0

    def test_estimate_v1_step_cost_with_condition(self) -> None:
        """Test V1 step cost estimation with condition."""
        mock_step = MagicMock()
        mock_step.output = "test_output"
        mock_step.task = "Test task"
        mock_step.condition = "some condition"

        estimate = self.estimator._estimate_v1_step_cost(mock_step, "gpt-4o")

        assert estimate.has_condition
        assert estimate.introspection_cost > 0

    @patch.object(CostEstimator, "_get_llm_estimation")
    def test_estimate_v2_step_cost(self, mock_llm_estimation: MagicMock) -> None:
        """Test V2 step cost estimation."""
        mock_llm_estimation.return_value = {
            "estimated_input_tokens": 1000,
            "estimated_output_tokens": 300,
            "number_of_llm_calls": 2,
            "reasoning": "Test reasoning",
        }

        step = LLMStep(step_name="test_step", task="Test task")
        estimate = self.estimator._estimate_v2_step_cost(step, "gpt-4o")

        assert isinstance(estimate, StepCostEstimate)
        assert estimate.step_name == "test_step"
        assert estimate.step_type == "LLMStep"
        assert estimate.estimated_input_tokens == 2000  # 1000 * 2 calls
        assert estimate.estimated_output_tokens == 600  # 300 * 2 calls
        assert estimate.estimated_cost > 0
        assert "Test reasoning" in estimate.explanation

    def test_estimate_v1_plan(self) -> None:
        """Test V1 plan cost estimation."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=[]),
            plan_inputs=[PlanInput(name="input1", description="Test input")],
            steps=[
                Step(task="Step 1", output="$output1"),
                Step(task="Step 2", output="$output2", condition="some condition"),
            ],
        )

        estimate = self.estimator._estimate_v1_plan(plan)

        assert isinstance(estimate, PlanCostEstimate)
        assert len(estimate.step_estimates) == 2
        assert estimate.total_estimated_cost > 0
        assert estimate.step_estimates[1].has_condition
        assert estimate.methodology
        assert estimate.limitations

    def test_estimate_v2_plan(self) -> None:
        """Test V2 plan cost estimation."""
        plan = (
            PlanBuilderV2("Test plan")
            .llm_step(step_name="step1", task="First task")
            .react_agent_step(step_name="step2", task="Second task", tools=["tool1"])
            .build()
        )

        with patch.object(self.estimator, "_get_llm_estimation") as mock_estimation:
            mock_estimation.return_value = {
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 300,
                "number_of_llm_calls": 2,
                "reasoning": "Test reasoning",
            }

            estimate = self.estimator._estimate_v2_plan(plan)

        assert isinstance(estimate, PlanCostEstimate)
        assert len(estimate.step_estimates) == 2
        assert estimate.total_estimated_cost > 0
        assert estimate.step_estimates[0].step_type == "LLMStep"
        assert estimate.step_estimates[1].step_type == "ReActAgentStep"

    def test_plan_estimate_v1(self) -> None:
        """Test main plan_estimate method with V1 plan."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=[]),
            steps=[Step(task="Test step", output="$output")],
        )

        estimate = self.estimator.plan_estimate(plan)
        assert isinstance(estimate, PlanCostEstimate)
        assert len(estimate.step_estimates) == 1

    def test_plan_estimate_v1_coverage_path(self) -> None:
        """Ensure V1 plan estimation path is fully covered."""
        from portia.plan import Plan

        plan = Plan(
            plan_context=PlanContext(query="V1 coverage test", tool_ids=[]),
            plan_inputs=[PlanInput(name="test_input", description="Coverage input")],
            steps=[
                Step(task="Coverage task", inputs=[], output="coverage_output"),
            ],
        )

        estimate = self.estimator.plan_estimate(plan)
        assert isinstance(estimate, PlanCostEstimate)
        assert estimate.total_estimated_cost > 0

    def test_plan_estimate_v2(self) -> None:
        """Test main plan_estimate method with V2 plan."""
        plan = PlanBuilderV2("Test plan").llm_step(step_name="step1", task="Test task").build()

        with patch.object(self.estimator, "_get_llm_estimation") as mock_estimation:
            mock_estimation.return_value = {
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 300,
                "number_of_llm_calls": 1,
                "reasoning": "Test reasoning",
            }

            estimate = self.estimator.plan_estimate(plan)

        assert isinstance(estimate, PlanCostEstimate)
        assert len(estimate.step_estimates) == 1

    @pytest.mark.asyncio
    async def test_aplan_estimate(self) -> None:
        """Test async plan estimation."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=[]),
            steps=[Step(task="Test step", output="$output")],
        )

        estimate = await self.estimator.aplan_estimate(plan)
        assert isinstance(estimate, PlanCostEstimate)

    def test_methodology_and_limitations(self) -> None:
        """Test methodology and limitations explanations."""
        methodology = self.estimator._get_methodology_explanation()
        assert "Cost estimation methodology" in methodology
        assert "LLM-driven estimation" in methodology

        limitations = self.estimator._get_limitations_explanation()
        assert "Limitations and assumptions" in limitations
        assert "Only includes LLM costs" in limitations


class TestStepCostEstimate:
    """Test cases for StepCostEstimate model."""

    def test_step_cost_estimate_creation(self) -> None:
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


class TestPlanCostEstimate:
    """Test cases for PlanCostEstimate model."""

    def test_plan_cost_estimate_creation(self) -> None:
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


class TestModelPricing:
    """Test cases for model pricing data."""

    def test_model_pricing_coverage(self) -> None:
        """Test that pricing data covers major models."""
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING
        assert "claude-3-5-sonnet-latest" in MODEL_PRICING
        assert "gemini-2.0-flash" in MODEL_PRICING

        for pricing in MODEL_PRICING.values():
            assert "input" in pricing
            assert "output" in pricing
            assert isinstance(pricing["input"], int | float)
            assert isinstance(pricing["output"], int | float)
            assert pricing["input"] > 0
            assert pricing["output"] > 0

    def test_pricing_reasonableness(self) -> None:
        """Test that pricing values are reasonable."""
        for pricing in MODEL_PRICING.values():
            assert pricing["output"] >= pricing["input"]
            assert pricing["input"] < 100
            assert pricing["output"] < 100
