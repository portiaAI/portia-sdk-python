"""Cost estimator for Portia plans.

This module provides functionality to estimate the cost of running a Portia plan,
focusing on LLM usage costs across execution agents, introspection agents, and LLM tools.

The cost estimator uses an LLM-driven approach to estimate token usage and costs
for each step in a plan, providing detailed breakdowns and explanations.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from portia.builder.llm_step import LLMStep
from portia.builder.react_agent_step import ReActAgentStep
from portia.builder.single_tool_agent_step import SingleToolAgentStep
from portia.config import Config
from portia.logger import logger
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan
from portia.token_check import estimate_tokens

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.builder.step_v2 import StepV2


MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o3-mini": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 5.00, "output": 15.00},
    # Anthropic
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.25, "output": 1.25},
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00},
    "claude-3-7-sonnet-latest": {"input": 3.00, "output": 15.00},
    # Google
    "gemini-2.5-flash-preview-04-17": {"input": 0.037, "output": 0.15},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.037, "output": 0.15},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.037, "output": 0.15},
    # MistralAI
    "mistral-large-latest": {"input": 3.00, "output": 9.00},
    # Grok
    "grok-4-0709": {"input": 10.00, "output": 30.00},
    "grok-3": {"input": 5.00, "output": 15.00},
    "grok-3-mini": {"input": 1.00, "output": 3.00},
}

DEFAULT_MODEL_PRICING = {"input": 5.00, "output": 15.00}


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
- Execution agents typically make 2-3 LLM calls: argument parsing, verification, and tool calling
- ReAct agents make multiple calls for reasoning and tool selection (typically 3-5 calls)
- Single tool agents make 1-2 calls for tool argument determination
- LLM steps make 1 direct call
- Complex tasks require longer responses
- Tool-calling steps have additional overhead for tool schemas

Provide your estimate as JSON with this exact structure:
{{
    "estimated_input_tokens": <number>,
    "estimated_output_tokens": <number>,
    "number_of_llm_calls": <number>,
    "reasoning": "<explanation of your estimation>"
}}
"""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the cost estimator.

        Args:
            config: Portia configuration to use. If None, uses default config.

        """
        self.config = config or Config.from_default()
        self.estimation_tool = LLMTool()

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

    def _estimate_v1_step_cost(self, step: object, model_name: str) -> StepCostEstimate:
        """Estimate cost for a V1 plan step."""
        step_name = f"Step {getattr(step, 'output', 'unnamed')}"
        step_type = "V1Step"
        task = getattr(step, "task", "No task description")

        input_tokens = estimate_tokens(task) + 500
        output_tokens = 200

        has_condition = hasattr(step, "condition") and getattr(step, "condition", None) is not None
        introspection_cost = 0.0

        if has_condition:
            introspection_cost = self._calculate_introspection_cost(model_name)

        cost = self._calculate_cost(input_tokens, output_tokens, model_name)
        total_cost = cost + introspection_cost

        return StepCostEstimate(
            step_name=step_name,
            step_type=step_type,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost=total_cost,
            cost_breakdown={"execution": cost, "introspection": introspection_cost},
            explanation=(
                f"Basic estimation for V1 step: {input_tokens} input + "
                f"{output_tokens} output tokens"
            ),
            has_condition=has_condition,
            introspection_cost=introspection_cost,
        )

    def _estimate_v2_step_cost(self, step: StepV2, model_name: str) -> StepCostEstimate:
        """Estimate cost for a V2 plan step using LLM-driven estimation."""
        step_name = step.step_name
        step_type = type(step).__name__

        task = self._extract_step_task(step)

        tools = self._get_step_tools(step)

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
            introspection_cost = self._calculate_introspection_cost(model_name)

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

    def _get_step_tools(self, step: StepV2) -> str:
        """Get tool information for a step."""
        if isinstance(step, ReActAgentStep):
            if step.tools:
                tool_names = []
                for tool in step.tools:
                    if isinstance(tool, str):
                        tool_names.append(tool)
                    else:
                        tool_names.append(tool.id)
                return f"Tools: {', '.join(tool_names)}"
            return "Tools: None"
        if isinstance(step, SingleToolAgentStep):
            tool_name = step.tool if isinstance(step.tool, str) else step.tool.id
            return f"Tool: {tool_name}"
        return "No tools"

    def _estimate_input_context(self, step: StepV2) -> int:
        """Estimate the input context size for a step."""
        base_context = 1000

        if hasattr(step, "inputs") and getattr(step, "inputs", None):
            inputs = getattr(step, "inputs", [])
            for _inp in inputs:
                base_context += 200

        return base_context

    def _get_llm_estimation(
        self, step_type: str, task: str, model_name: str, tools: str, input_context_tokens: int
    ) -> dict[str, Any]:
        """Use LLM to estimate token usage for a step."""
        from portia.end_user import EndUser
        from portia.plan import Plan, PlanContext
        from portia.plan_run import PlanRun, PlanRunState
        from portia.tool import ToolRunContext

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

            if isinstance(response, str):
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)

            return self._get_fallback_estimation(step_type)

        except (json.JSONDecodeError, ValueError, KeyError, Exception) as e:
            logger().warning(f"LLM estimation failed: {e}, using fallback")
            return self._get_fallback_estimation(step_type)

    def _get_fallback_estimation(self, step_type: str) -> dict[str, Any]:
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
                "number_of_llm_calls": 2,
                "reasoning": "Fallback: Tool agent with argument parsing",
            },
        }

        return estimations.get(
            step_type,
            {
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 300,
                "number_of_llm_calls": 2,
                "reasoning": f"Fallback: Unknown step type {step_type}",
            },
        )

    def _calculate_introspection_cost(self, model_name: str) -> float:
        """Calculate the cost of running the introspection agent."""
        input_tokens = 800
        output_tokens = 100

        return self._calculate_cost(input_tokens, output_tokens, model_name)

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate the cost in USD for the given token usage."""
        pricing = MODEL_PRICING.get(model_name, DEFAULT_MODEL_PRICING)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _get_methodology_explanation(self) -> str:
        """Get explanation of the cost estimation methodology."""
        return """
Cost estimation methodology:
1. Identifies each step type and its LLM usage patterns
2. Uses LLM-driven estimation to predict token usage based on step complexity
3. Accounts for multiple LLM calls per step (execution agents make 2-3 calls)
4. Includes introspection agent costs for conditional steps
5. Applies current model pricing per million tokens
6. Estimates both input (prompts) and output (responses) token usage

This approach adapts to changes in the SDK without requiring manual updates
to complex calculation methods.
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
