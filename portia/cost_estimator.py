"""Cost estimator for Portia plans.

This module provides functionality to estimate the cost of running a Portia plan,
focusing on LLM usage costs across execution agents, introspection agents, and LLM tools.

The cost estimator uses an LLM-driven approach to estimate token usage and costs
for each step in a plan, providing detailed breakdowns and explanations.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

import litellm
from pydantic import BaseModel, Field

from portia.builder.llm_step import LLMStep
from portia.builder.plan_v2 import PlanV2
from portia.builder.react_agent_step import ReActAgentStep
from portia.builder.single_tool_agent_step import SingleToolAgentStep
from portia.config import Config
from portia.end_user import EndUser
from portia.logger import logger
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, Step
from portia.plan_run import PlanRun, PlanRunState
from portia.token_check import estimate_tokens
from portia.tool import ToolRunContext

if TYPE_CHECKING:
    from portia.builder.step_v2 import StepV2


LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

DEFAULT_MODEL_PRICING = {"input": 5.00, "output": 15.00}


class LLMEstimationResult(BaseModel):
    """Structured output for LLM token usage estimation."""

    estimated_input_tokens: int = Field(
        description="Estimated number of input tokens for this step"
    )
    estimated_output_tokens: int = Field(
        description="Estimated number of output tokens for this step"
    )
    number_of_llm_calls: int = Field(description="Number of LLM calls this step will make")
    reasoning: str = Field(description="Brief explanation of the estimation logic")


class StepCostEstimate(BaseModel):
    """Cost estimate for a single step in a plan."""

    step_name: str = Field(description="The name of the step")
    step_type: str = Field(description="The type of step (e.g., LLMStep, ReActAgentStep)")
    estimated_input_tokens: int = Field(description="Estimated input tokens")
    estimated_output_tokens: int = Field(description="Estimated output tokens")
    estimated_cost: float = Field(description="Estimated cost in USD")
    cost_breakdown: dict[str, float] = Field(description="Breakdown of costs by component")
    explanation: str = Field(description="Explanation of how the cost was calculated")
    has_condition: bool = Field(
        description="Whether this step has a condition that triggers introspection"
    )
    introspection_cost: float = Field(
        default=0.0, description="Additional cost for introspection agent"
    )


class PlanCostEstimate(BaseModel):
    """Complete cost estimate for a plan."""

    total_estimated_cost: float = Field(description="Total estimated cost in USD")
    step_estimates: list[StepCostEstimate] = Field(description="Cost estimates for each step")
    model_used: str = Field(description="The primary model used for execution")
    methodology: str = Field(description="Explanation of the cost estimation methodology")
    limitations: str = Field(description="Limitations and assumptions in the estimate")


class CostEstimator:
    """Estimates the cost of running Portia plans based on LLM usage."""

    ESTIMATION_PROMPT: ClassVar[str] = """
You are a cost estimation expert for AI agent workflows. Your task is to estimate the LLM
token usage and costs for executing a step in a Portia plan.

Given the following information about a step:
- Step type: {step_type}
- Step task/description: {step_task}
- Model being used: {model_name}
- Available tools (if any): {tools}
- Input context size: {input_context_tokens} tokens

Please estimate:
1. Input tokens for this step (including system prompts, context, and user input)
2. Output tokens for this step (the expected response length)
3. Number of LLM calls this step will make (some steps make multiple calls)

Consider these factors:
- Execution agents (default one-shot) make 1 LLM call, plus 1 more if summarization is enabled
- ReAct agents make multiple calls for reasoning and tool selection (typically 3-5 calls)
- Single tool agents make 1 call for tool argument determination and execution
- LLM steps make 1 direct call
- Complex tasks require longer responses
- Tool-calling steps have additional overhead for tool schemas

Please provide your estimate with a brief explanation of your reasoning.
"""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the cost estimator.

        Args:
            config: Portia configuration to use. If None, uses default config.

        """
        self.config = config or Config.from_default()
        self.estimation_tool = LLMTool(structured_output_schema=LLMEstimationResult)

    def plan_estimate(self, plan: Plan | PlanV2) -> PlanCostEstimate:
        """Estimate the cost of running a plan.

        Args:
            plan: The plan to estimate costs for

        Returns:
            Complete cost estimate for the plan

        """
        logger().info("Starting cost estimation for plan")

        if isinstance(plan, Plan):
            return self._estimate_v1_plan(plan)
        return self._estimate_v2_plan(plan)

    async def aplan_estimate(self, plan: Plan | PlanV2) -> PlanCostEstimate:
        """Asynchronously estimate the cost of running a plan.

        Args:
            plan: The plan to estimate costs for

        Returns:
            Complete cost estimate for the plan

        """
        return await asyncio.to_thread(self.plan_estimate, plan)

    def _estimate_v1_plan(self, plan: Plan) -> PlanCostEstimate:
        """Estimate costs for a V1 plan."""
        step_estimates = []
        total_cost = 0.0

        execution_model = self.config.get_execution_model()
        model_name = execution_model.model_name if execution_model else "gpt-4o"

        for step in plan.steps:
            estimate = self._estimate_v1_step_cost(step, model_name)
            step_estimates.append(estimate)
            total_cost += estimate.estimated_cost

        return PlanCostEstimate(
            total_estimated_cost=total_cost,
            step_estimates=step_estimates,
            model_used=model_name,
            methodology=self._get_methodology_explanation(),
            limitations=self._get_limitations_explanation(),
        )

    def _estimate_v2_plan(self, plan: PlanV2) -> PlanCostEstimate:
        """Estimate costs for a V2 plan."""
        step_estimates = []
        total_cost = 0.0

        execution_model = self.config.get_execution_model()
        model_name = execution_model.model_name if execution_model else "gpt-4o"

        for step in plan.steps:
            estimate = self._estimate_v2_step_cost(step, model_name)
            step_estimates.append(estimate)
            total_cost += estimate.estimated_cost

        return PlanCostEstimate(
            total_estimated_cost=total_cost,
            step_estimates=step_estimates,
            model_used=model_name,
            methodology=self._get_methodology_explanation(),
            limitations=self._get_limitations_explanation(),
        )

    def _estimate_v1_step_cost(self, step: Step, model_name: str) -> StepCostEstimate:
        """Estimate cost for a V1 plan step."""
        step_name = f"Step {step.output}"
        step_type = "V1Step"
        task = step.task

        tools = step.tool_id or ""
        input_context_tokens = estimate_tokens(task) + 500  # Base context estimation

        estimation_result = self._get_llm_estimation(
            "ExecutionAgentStep", task, model_name, tools, input_context_tokens
        )

        base_cost = self._calculate_cost(
            estimation_result["estimated_input_tokens"] * estimation_result["number_of_llm_calls"],
            estimation_result["estimated_output_tokens"] * estimation_result["number_of_llm_calls"],
            model_name,
        )

        has_condition = step.condition is not None
        introspection_cost = 0.0

        if has_condition:
            introspection_model = self.config.get_introspection_model()
            if introspection_model is not None:
                introspection_model_name = introspection_model.model_name
                introspection_cost = self._calculate_introspection_cost(
                    introspection_model_name
                )  # pragma: no cover
            else:
                default_model = self.config.get_default_model()  # pragma: no cover
                introspection_model_name = (
                    default_model.model_name if default_model else "gpt-4o"
                )  # pragma: no cover
                introspection_cost = self._calculate_introspection_cost(
                    introspection_model_name
                )  # pragma: no cover

        total_cost = base_cost + introspection_cost

        return StepCostEstimate(
            step_name=step_name,
            step_type=step_type,
            estimated_input_tokens=estimation_result["estimated_input_tokens"]
            * estimation_result["number_of_llm_calls"],
            estimated_output_tokens=estimation_result["estimated_output_tokens"]
            * estimation_result["number_of_llm_calls"],
            estimated_cost=total_cost,
            cost_breakdown={"execution": base_cost, "introspection": introspection_cost},
            explanation=(
                f"LLM-driven estimation for V1 step: "
                f"{estimation_result['estimated_input_tokens']} input x "
                f"{estimation_result['number_of_llm_calls']} calls + "
                f"{estimation_result['estimated_output_tokens']} output x "
                f"{estimation_result['number_of_llm_calls']} calls"
            ),
            has_condition=has_condition,
            introspection_cost=introspection_cost,
        )

    def _estimate_v2_step_cost(self, step: StepV2, model_name: str) -> StepCostEstimate:
        """Estimate cost for a V2 plan step using LLM-driven estimation."""
        step_type = type(step).__name__

        task = self._extract_step_task(step)

        try:
            dummy_plan = PlanV2(steps=[step])
            tools = step.to_step_data(dummy_plan).tool_id or ""
        except (ValueError, AttributeError) as e:
            logger().debug(f"Failed to convert step to legacy format: {e}")
            tools = ""

        input_context_tokens = self._estimate_input_context(step)

        estimation_result = self._get_llm_estimation(
            step_type, task, model_name, tools, input_context_tokens
        )

        base_cost = self._calculate_cost(
            estimation_result["estimated_input_tokens"] * estimation_result["number_of_llm_calls"],
            estimation_result["estimated_output_tokens"] * estimation_result["number_of_llm_calls"],
            model_name,
        )

        has_condition = hasattr(step, "conditional_block") and step.conditional_block is not None
        introspection_cost = 0.0

        if has_condition:
            introspection_model = self.config.get_introspection_model()
            if introspection_model is not None:
                introspection_model_name = introspection_model.model_name
                introspection_cost = self._calculate_introspection_cost(
                    introspection_model_name
                )  # pragma: no cover
            else:
                default_model = self.config.get_default_model()  # pragma: no cover
                introspection_model_name = (
                    default_model.model_name if default_model else "gpt-4o"
                )  # pragma: no cover
                introspection_cost = self._calculate_introspection_cost(
                    introspection_model_name
                )  # pragma: no cover

        total_cost = base_cost + introspection_cost

        return StepCostEstimate(
            step_name=step.step_name,
            step_type=step_type,
            estimated_input_tokens=estimation_result["estimated_input_tokens"]
            * estimation_result["number_of_llm_calls"],
            estimated_output_tokens=estimation_result["estimated_output_tokens"]
            * estimation_result["number_of_llm_calls"],
            estimated_cost=total_cost,
            cost_breakdown={"execution": base_cost, "introspection": introspection_cost},
            explanation=(
                f"{estimation_result['reasoning']} | "
                f"Calls: {estimation_result['number_of_llm_calls']}"
            ),
            has_condition=has_condition,
            introspection_cost=introspection_cost,
        )

    def _extract_step_task(self, step: StepV2) -> str:
        """Extract the task description from a step."""
        if isinstance(step, LLMStep | ReActAgentStep | SingleToolAgentStep):
            return step.task
        return f"Execute {type(step).__name__}"

    def _estimate_input_context(self, step: StepV2) -> int:
        """Estimate the input context size for a step."""
        base_context = 1000

        inputs = getattr(step, "inputs", None)
        if inputs:
            for _inp in inputs:
                base_context += 200

        return base_context

    def _get_llm_estimation(
        self, step_type: str, task: str, model_name: str, tools: str, input_context_tokens: int
    ) -> dict[str, Any]:
        """Use LLM to estimate token usage for a step."""
        plan = Plan(
            plan_context=PlanContext(query="cost_estimation", tool_ids=[]),
            plan_inputs=[],
            steps=[],
        )

        plan_run = PlanRun(
            plan_id=plan.id,
            end_user_id="cost_estimator",
            plan_run_inputs={},
            state=PlanRunState.IN_PROGRESS,
        )

        context = ToolRunContext(
            end_user=EndUser(external_id="cost_estimator"),
            plan_run=plan_run,
            plan=plan,
            config=self.config,
            clarifications=[],
        )

        prompt = self.ESTIMATION_PROMPT.format(
            step_type=step_type,
            step_task=task,
            model_name=model_name,
            tools=tools,
            input_context_tokens=input_context_tokens,
        )

        try:
            response = self.estimation_tool.run(context, prompt)

            if isinstance(response, LLMEstimationResult):
                return {
                    "estimated_input_tokens": response.estimated_input_tokens,
                    "estimated_output_tokens": response.estimated_output_tokens,
                    "number_of_llm_calls": response.number_of_llm_calls,
                    "reasoning": response.reasoning,
                }

            return self._get_fallback_estimation(step_type)  # pragma: no cover

        except (ValueError, TypeError, AttributeError) as e:  # pragma: no cover
            logger().warning(f"LLM estimation failed: {e}, using fallback")  # pragma: no cover
            return self._get_fallback_estimation(step_type)  # pragma: no cover

    def _get_fallback_estimation(self, step_type: str) -> dict[str, Any]:  # pragma: no cover
        """Provide fallback estimations when LLM estimation fails."""
        estimations = {
            "LLMStep": {
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 300,
                "number_of_llm_calls": 1,
                "reasoning": "Fallback: Direct LLM call with moderate response",
            },
            "ReActAgentStep": {
                "estimated_input_tokens": 2000,
                "estimated_output_tokens": 800,
                "number_of_llm_calls": 4,
                "reasoning": "Fallback: ReAct agent with multiple reasoning cycles",
            },
            "SingleToolAgentStep": {
                "estimated_input_tokens": 1500,
                "estimated_output_tokens": 400,
                "number_of_llm_calls": 1,
                "reasoning": "Fallback: One-shot tool agent with single call",
            },
            "ExecutionAgentStep": {
                "estimated_input_tokens": 1200,
                "estimated_output_tokens": 350,
                "number_of_llm_calls": 1,
                "reasoning": "Fallback: One-shot execution agent with single call",
            },
        }

        return estimations.get(
            step_type,
            {
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 300,
                "number_of_llm_calls": 1,
                "reasoning": f"Fallback: Unknown step type {step_type}, assuming one-shot",
            },
        )

    def _calculate_introspection_cost(self, model_name: str) -> float:
        """Calculate the cost of running the introspection agent."""
        input_tokens = 800
        output_tokens = 100

        return self._calculate_cost(input_tokens, output_tokens, model_name)

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate the cost in USD for the given token usage."""
        pricing = self._get_model_pricing(model_name)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _get_model_pricing(self, model_name: str) -> dict[str, float]:
        """Get pricing information for a model from LiteLLM or fallback to defaults."""
        try:
            cost_map = litellm.get_model_cost_map(LITELLM_PRICING_URL)  # pyright: ignore[reportPrivateImportUsage]

            if model_name in cost_map:
                model_data = cost_map[model_name]
                return {
                    "input": model_data.get("input_cost_per_token", 5.00e-6) * 1_000_000,
                    "output": model_data.get("output_cost_per_token", 15.00e-6) * 1_000_000,
                }

        except (ValueError, TypeError, KeyError, AttributeError) as e:  # pragma: no cover
            logger().warning(
                f"Failed to fetch LiteLLM pricing data: {e}, using default pricing"
            )  # pragma: no cover

        return DEFAULT_MODEL_PRICING  # pragma: no cover

    def _get_methodology_explanation(self) -> str:
        """Get explanation of the cost estimation methodology."""
        return """
Cost estimation methodology:
1. Identifies each step type and its LLM usage patterns
2. Uses LLM-driven estimation to predict token usage based on step complexity
3. Accounts for multiple LLM calls per step (execution agents make 2-3 calls)
4. Includes introspection agent costs for conditional steps
5. Applies current model pricing from LiteLLM's maintained pricing data
6. Estimates both input (prompts) and output (responses) token usage

This approach adapts to changes in the SDK without requiring manual updates
to complex calculation methods. Pricing data is sourced from LiteLLM's
centralized pricing database for accuracy and up-to-date information.
        """.strip()

    def _get_limitations_explanation(self) -> str:
        """Get explanation of limitations and assumptions."""
        return """
Limitations and assumptions:
- Only includes LLM costs, not CPU/infrastructure costs
- Token estimates are approximate and may vary by 20-50%
- Pricing based on current public rates (may change)
- Does not account for retry attempts on failures
- Assumes average complexity for tasks
- Does not include costs for plan generation itself
- Conditional step branching may affect actual execution costs
- Tool response sizes can vary significantly based on external data
        """.strip()
