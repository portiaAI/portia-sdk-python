"""Cost Estimation for Portia Plans.

This module provides a comprehensive framework for estimating the monetary cost
 of executing a plan within the Portia SDK.
It leverages a generative model (`CostEstimator.estimator`) to intelligently
predict token usage for various components of a plan's execution, including:

- **Execution Agent:** The main LLM responsible for carrying out a plan step.
- **Introspection Agent:** A dedicated LLM for evaluating conditional logic in a plan.
- **LLM Tool Calls:** Direct usage of generative models as tools within a step.
- **Other Operations:** Ad-hoc operations like summarization of large inputs or outputs.

The `CostEstimator` class uses these token estimates, combined with model-specific
pricing data (`MODEL_PRICING`), to calculate a total estimated cost for an entire plan,
 providing a detailed breakdown for each step.

## Key Features

- **Comprehensive Breakdown:** Provides a step-by-step breakdown of costs,
 including separate estimates for different agent and tool interactions.
- **LLM-Based Estimation:** Uses a dedicated LLM to make informed predictions about token usage,
 leading to more accurate estimates than simple hardcoded values.
- **Configurable Pricing:** Supports adding and managing pricing for different generative models.
- **Detailed Logging:** Logs a full summary of the estimation process,
 from individual step breakdowns to a final total.

## Limitations

- **Approximation Only:** All estimates are predictions, not actual token counts.
- **Real costs may vary.
- **External Costs:** Does not account for infrastructure costs (CPU, memory, etc).
- **Dynamic Behavior:** The actual execution path of a plan,
and therefore its cost, can change based on runtime data and conditional logic.

"""

from typing import Any

from pydantic import BaseModel, Field

from portia.config import Config
from portia.errors import InvalidAgentOutputError, ToolHardError
from portia.logger import logger
from portia.model import GenerativeModel, Message
from portia.plan import Plan, Step
from portia.portia import Portia


class TokenCount(BaseModel):
    """A structured representation of token counts."""

    input_tokens: int = Field(..., description="The number of tokens in the input prompt.")
    output_tokens: int = Field(..., description="The number of tokens in the generated response.")


class StepCostBreakdown(BaseModel):
    """Detailed breakdown of costs for a step."""

    execution_agent: dict[str, Any] = Field(
        default_factory=dict, description="Cost of execution agent"
    )
    introspection_agent: dict[str, Any] = Field(
        default_factory=dict, description="Cost of introspection agent"
    )
    llm_tools: list[dict[str, Any]] = Field(
        default_factory=list, description="Cost of LLM tool calls"
    )
    other_costs: list[dict[str, Any]] = Field(
        default_factory=list, description="Other miscellaneous costs"
    )


# Cost per 1K tokens (USD) for various models
MODEL_PRICING = {
    "google/gemini-2.0-flash": {
        "input": 0.10 / 1000,
        "output": 0.40 / 1000,
    },
    "google/gemini-2.0-flash-lite": {
        "input": 0.10 / 1000,
        "output": 0.40 / 1000,
    },
    "google/gemini-1.5-flash": {
        "input": 0.10 / 1000,
        "output": 0.40 / 1000,
    },
    # Add more models as needed
    "default": {
        "input": 0.10 / 1000,
        "output": 0.40 / 1000,
    },
}


class CostEstimator:
    """Estimates the monetary cost of executing a plan within the Portia SDK.

    This class provides a comprehensive framework for calculating plan costs...
    ... (add the rest of the detailed docstring here)
    """

    def __init__(
        self, portia: Portia | None = None, estimator_model: str = "google/gemini-2.0-flash-lite"
    ) -> None:
        """Initialize the CostEstimator instance.

        Args:
            portia: The main Portia SDK instance, or None to use default config.
            estimator_model: The name of the model to use for cost estimation queries.

        """
        self.model_costs = MODEL_PRICING
        self.estimator_model = estimator_model
        config = portia.config if portia else Config.from_default()
        # ✅ This will return a fully configured GenerativeModel
        self.estimator: GenerativeModel | None = config.get_generative_model(self.estimator_model)

    def plan_estimate(self, plan: Plan) -> dict[str, Any]:
        """Estimate the total cost of running a plan.

        Args:
            plan: The execution plan to estimate costs for

        Returns:
            Dictionary containing total cost, step-by-step breakdown, and methodology explanation

        """
        logger().info("=" * 50)
        logger().info("STARTING PLAN COST ESTIMATION")
        logger().info("=" * 50)

        step_costs = []
        total_cost = 0.0

        logger().info(f"Analyzing plan with {len(plan.steps)} steps...")
        logger().info("")

        for i, step in enumerate(plan.steps):
            logger().info(f"Estimating cost for Step {i + 1}...")
            logger().info(f"Step description: {step!s}")

            step_cost, details = self._estimate_step_cost_comprehensive(step, i)

            step_data = {
                "step_number": i + 1,
                "step_description": str(step),
                "total_cost": step_cost,
                "total_cost_formatted": f"${step_cost:.6f}",
                "breakdown": details,
            }
            step_costs.append(step_data)
            total_cost += step_cost

            # Log detailed breakdown for this step
            self._log_step_breakdown(step_data)
            logger().info("")

        result = {
            "total_cost": total_cost,
            "total_cost_formatted": f"${total_cost:.6f}",
            "steps": step_costs,
            "methodology": self._get_methodology_explanation(),
            "limitations": self._get_limitations(),
            "currency": "USD",
        }

        # Log final summary
        self._log_final_summary(result)

        return result

    def _estimate_step_cost_comprehensive(
        self, step: Step, step_index: int
    ) -> tuple[float, dict[str, Any]]:
        """Comprehensively estimate the cost of a single step including all components."""
        breakdown = StepCostBreakdown()
        total_step_cost = 0.0

        # 1. Estimate execution agent cost
        execution_cost, execution_details = self._estimate_execution_agent_cost(step, step_index)
        breakdown.execution_agent = execution_details
        total_step_cost += execution_cost

        # 2. Estimate introspection agent cost (if step has conditions)
        if self._step_has_condition(step):
            introspection_cost, introspection_details = self._estimate_introspection_agent_cost(
                step
            )
            breakdown.introspection_agent = introspection_details
            total_step_cost += introspection_cost

        # 3. Estimate LLM tool costs
        llm_tool_costs = self._estimate_llm_tool_costs(step)
        breakdown.llm_tools = llm_tool_costs
        for tool_cost in llm_tool_costs:
            total_step_cost += tool_cost.get("total_cost", 0.0)

        # 4. Estimate other costs (summarization, etc.)
        other_costs = self._estimate_other_costs(step)
        breakdown.other_costs = other_costs
        for other_cost in other_costs:
            total_step_cost += other_cost.get("total_cost", 0.0)

        return total_step_cost, breakdown.dict()

    def _estimate_execution_agent_cost(
        self, step: Step, step_index: int
    ) -> tuple[float, dict[str, Any]]:
        """Estimate the cost of running the execution agent for this step."""
        model_name = getattr(step, "model", self.estimator_model)
        pricing = self.model_costs.get(model_name, self.model_costs["default"])

        logger().info(f"  → Estimating execution agent cost (model: {model_name})")

        prompt = f"""
You are estimating token usage for an execution agent processing a step in a plan.

Step Information:
- Step number: {step_index + 1}
- Description: {getattr(step, 'description', str(step))}
- Task: {getattr(step, 'task', 'Not specified')}
- Tool ID: {getattr(step, 'tool_id', 'llm_tool')}
- Inputs: {getattr(step, 'inputs', {})}
- Output variable: {getattr(step, 'output', 'Not specified')}
- Model: {model_name}

The execution agent will:
1. Process the step description and task
2. Consider the inputs and context
3. Execute the required action
4. Generate appropriate output

Estimate the token usage for the EXECUTION AGENT specifically.
Consider that the prompt will include system instructions, step context, and input data.
The response will include the agent's reasoning and final output.

Return only JSON with keys: input_tokens, output_tokens (integers only).
"""

        try:
            if self.estimator is None:
                raise RuntimeError("Estimator not initialized")
            response: TokenCount = self.estimator.get_structured_response(
                messages=[Message(role="user", content=prompt)], schema=TokenCount
            )

            input_tokens = response.input_tokens
            output_tokens = response.output_tokens

            input_cost = (input_tokens / 1000.0) * pricing["input"]
            output_cost = (output_tokens / 1000.0) * pricing["output"]
            total_cost = input_cost + output_cost

            logger().info(
                f"    Execution agent: {input_tokens} input + "
                f"{output_tokens} output tokens = ${total_cost:.6f}"
            )

            details = {
                "component": "execution_agent",
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            }
        except (ToolHardError, InvalidAgentOutputError) as e:
            logger().warning(f"    Error estimating execution agent cost: {e}")
            return 0.0, {"component": "execution_agent", "error": str(e), "total_cost": 0.0}
        else:
            return total_cost, details

    def _estimate_introspection_agent_cost(self, step: Step) -> tuple[float, dict[str, Any]]:
        """Estimate the cost of running the introspection agent for conditional steps."""
        model_name = getattr(step, "model", self.estimator_model)
        pricing = self.model_costs.get(model_name, self.model_costs["default"])

        logger().info("  → Estimating introspection agent cost (conditional step)")

        prompt = f"""
You are estimating token usage for an introspection agent evaluating a conditional step.

Step Information:
- Description: {getattr(step, 'description', str(step))}
- Condition: {getattr(step, 'condition', 'Has conditional logic')}
- Context: {getattr(step, 'inputs', {})}

The introspection agent will:
1. Analyze the current state and context
2. Evaluate the condition(s)
3. Determine if the step should be executed
4. Provide reasoning for the decision

This is typically a smaller, focused evaluation task.

Return only JSON with keys: input_tokens, output_tokens (integers only).
"""

        try:
            if self.estimator is None:
                raise RuntimeError("Estimator not initialized")
            response: TokenCount = self.estimator.get_structured_response(
                messages=[Message(role="user", content=prompt)], schema=TokenCount
            )

            input_tokens = response.input_tokens
            output_tokens = response.output_tokens

            input_cost = (input_tokens / 1000.0) * pricing["input"]
            output_cost = (output_tokens / 1000.0) * pricing["output"]
            total_cost = input_cost + output_cost

            logger().info(
                f"    Introspection agent: {input_tokens} input + "
                f"{output_tokens} output tokens = ${total_cost:.6f}"
            )

            details = {
                "component": "introspection_agent",
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            }

        except (ToolHardError, InvalidAgentOutputError) as e:
            logger().warning(f"    Error estimating introspection agent cost: {e}")
            return 0.0, {"component": "introspection_agent", "error": str(e), "total_cost": 0.0}
        else:
            return total_cost, details

    def _estimate_llm_tool_costs(self, step: Step) -> list[dict[str, Any]]:
        """Estimate costs for LLM tool calls within the step."""
        tool_costs = []
        tool_id = getattr(step, "tool_id", None)

        # If this step uses llm_tool or similar LLM-based tools
        if tool_id == "llm_tool" or self._is_llm_tool(step):
            logger().info(f"  → Estimating LLM tool cost (tool: {tool_id})")

            model_name = getattr(step, "model", self.estimator_model)
            pricing = self.model_costs.get(model_name, self.model_costs["default"])

            prompt = f"""
You are estimating token usage for LLM tool calls within a step.

Step Information:
- Tool ID: {tool_id}
- Description: {getattr(step, 'description', str(step))}
- Task: {getattr(step, 'task', 'Not specified')}
- Inputs: {getattr(step, 'inputs', {})}

This step involves direct LLM tool usage for processing, generation, or analysis.
The LLM tool will receive the task prompt and generate a response.

Estimate tokens for the LLM tool call itself (separate from execution agent overhead).

Return only JSON with keys: input_tokens, output_tokens (integers only).
"""

            try:
                if self.estimator is None:
                    raise RuntimeError("Estimator not initialized")
                response: TokenCount = self.estimator.get_structured_response(
                    messages=[Message(role="user", content=prompt)], schema=TokenCount
                )

                input_tokens = response.input_tokens
                output_tokens = response.output_tokens

                input_cost = (input_tokens / 1000.0) * pricing["input"]
                output_cost = (output_tokens / 1000.0) * pricing["output"]
                total_cost = input_cost + output_cost

                logger().info(
                    f"    LLM tool ({tool_id}): {input_tokens} input + "
                    f"{output_tokens} output tokens = ${total_cost:.6f}"
                )

                tool_cost = {
                    "component": "llm_tool",
                    "tool_id": tool_id,
                    "model": model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                }

                tool_costs.append(tool_cost)

            except (ToolHardError, InvalidAgentOutputError) as e:
                logger().warning(f"    Error estimating LLM tool cost: {e}")
                tool_costs.append(
                    {
                        "component": "llm_tool",
                        "tool_id": tool_id,
                        "error": str(e),
                        "total_cost": 0.0,
                    }
                )

        return tool_costs

    def _estimate_other_costs(self, step: Step) -> list[dict[str, Any]]:
        """Estimate costs for other operations like summarization."""
        other_costs = []

        # Check if summarization might be involved
        if self._might_need_summarization(step):
            logger().info("  → Estimating summarization cost")

            model_name = getattr(step, "model", self.estimator_model)
            pricing = self.model_costs.get(model_name, self.model_costs["default"])

            prompt = f"""
You are estimating token usage for summarization operations in a step.

Step Information:
- Description: {getattr(step, 'description', str(step))}
- Context: This step may require summarization of large inputs or outputs

Summarization typically involves:
1. Processing large input text
2. Generating concise summaries
3. Maintaining key information

Estimate moderate token usage for summarization operations.

Return only JSON with keys: input_tokens, output_tokens (integers only).
"""

            try:
                if self.estimator is None:
                    raise RuntimeError("Estimator not initialized")
                response: TokenCount = self.estimator.get_structured_response(
                    messages=[Message(role="user", content=prompt)], schema=TokenCount
                )

                input_tokens = response.input_tokens
                output_tokens = response.output_tokens

                input_cost = (input_tokens / 1000.0) * pricing["input"]
                output_cost = (output_tokens / 1000.0) * pricing["output"]
                total_cost = input_cost + output_cost

                logger().info(
                    f"    Summarization: {input_tokens} input + "
                    f"{output_tokens} output tokens = ${total_cost:.6f}"
                )

                other_cost = {
                    "component": "summarization",
                    "model": model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                }

                other_costs.append(other_cost)

            except (ToolHardError, InvalidAgentOutputError) as e:
                logger().warning(f"    Error estimating summarization cost: {e}")

        return other_costs

    def _step_has_condition(self, step: Step) -> bool:
        """Check if a step has conditional logic requiring introspection agent."""
        return hasattr(step, "condition") or "condition" in str(step).lower()

    def _is_llm_tool(self, step: Step) -> bool:
        """Check if a step uses LLM-based tools."""
        tool_id = getattr(step, "tool_id", "")
        description = getattr(step, "description", "")
        return (
            "llm" in tool_id.lower()
            or "generate" in description.lower()
            or "analyze" in description.lower()
        )

    def _might_need_summarization(self, step: Step) -> bool:
        """Check if a step might require summarization."""
        description = getattr(step, "description", "").lower()
        task = getattr(step, "task", "").lower()
        return "large" in description or "summary" in description or "summarize" in task

    def _get_methodology_explanation(self) -> str:
        """Provide explanation of how costs are calculated."""
        return """
Cost estimation methodology:
1. Execution Agent: Estimates tokens for the main agent processing each step
2. Introspection Agent: Estimates tokens for conditional evaluation (if applicable)
3. LLM Tools: Estimates tokens for direct LLM tool usage within steps
4. Other Operations: Estimates tokens for summarization and other LLM operations
5. Final cost calculated using model-specific pricing per 1K tokens

Each component is estimated using an LLM that analyzes the step context and provides
token count estimates based on typical usage patterns.
"""

    def _get_limitations(self) -> str:
        """Provide limitations of the cost estimation."""
        return """
Limitations:
- Estimates are approximations based on typical usage patterns
- Actual costs may vary based on runtime conditions and data size
- Does not include CPU, memory, or infrastructure costs
- Token estimates are based on predictions, not actual measurement
- Complex conditional logic may result in different execution paths
"""

    def get_model_pricing(self) -> dict[str, dict[str, float]]:
        """Return current model pricing information."""
        return self.model_costs

    def add_model_pricing(
        self, model_name: str, input_cost_per_1k: float, output_cost_per_1k: float
    ) -> None:
        """Add pricing information for a new model."""
        self.model_costs[model_name] = {
            "input": input_cost_per_1k / 1000,
            "output": output_cost_per_1k / 1000,
        }
        logger().info(f"Added pricing for model {model_name}")

    def _log_step_breakdown(self, step_data: dict[str, Any]) -> None:
        """Log detailed breakdown for a single step."""
        logger().info(
            f"Step {step_data['step_number']} Total Cost: {step_data['total_cost_formatted']}"
        )

        breakdown = step_data["breakdown"]

        # Execution agent cost
        if (
            breakdown.get("execution_agent")
            and breakdown["execution_agent"].get("total_cost", 0) > 0
        ):
            exec_cost = breakdown["execution_agent"]
            logger().info(f"  ├─ Execution Agent: ${exec_cost['total_cost']:.6f}")
            logger().info(
                f"  │  └─ Tokens: {exec_cost.get('input_tokens', 0)} input, "
                f"{exec_cost.get('output_tokens', 0)} output"
            )

        # Introspection agent cost
        if (
            breakdown.get("introspection_agent")
            and breakdown["introspection_agent"].get("total_cost", 0) > 0
        ):
            intro_cost = breakdown["introspection_agent"]
            logger().info(f"  ├─ Introspection Agent: ${intro_cost['total_cost']:.6f}")
            logger().info(
                f"  │  └─ Tokens: {intro_cost.get('input_tokens', 0)} input, "
                f"{intro_cost.get('output_tokens', 0)} output"
            )

        # LLM tool costs
        if breakdown.get("llm_tools"):
            for tool_cost in breakdown["llm_tools"]:
                if tool_cost.get("total_cost", 0) > 0:
                    logger().info(
                        f"  ├─ LLM Tool ({tool_cost.get('tool_id', 'unknown')}): "
                        f"${tool_cost['total_cost']:.6f}"
                    )
                    logger().info(
                        f"  │  └─ Tokens: {tool_cost.get('input_tokens', 0)} input, "
                        f"{tool_cost.get('output_tokens', 0)} output"
                    )

        # Other costs
        if breakdown.get("other_costs"):
            for other_cost in breakdown["other_costs"]:
                if other_cost.get("total_cost", 0) > 0:
                    logger().info(
                        f"  ├─ {other_cost.get('component', 'Other').title()}: "
                        f"${other_cost['total_cost']:.6f}"
                    )
                    logger().info(
                        f"  │  └─ Tokens: {other_cost.get('input_tokens', 0)} input, "
                        f" {other_cost.get('output_tokens', 0)} output"
                    )

    def _log_final_summary(self, result: dict[str, Any]) -> None:
        """Log the final cost estimation summary."""
        logger().info("=" * 50)
        logger().info("COST ESTIMATION COMPLETE")
        logger().info("=" * 50)
        logger().info(
            f"TOTAL ESTIMATED COST: {result['total_cost_formatted']} {result['currency']}"
        )
        logger().info(f"Number of steps analyzed: {len(result['steps'])}")
        logger().info("")

        logger().info("STEP SUMMARY:")
        logger().info("-" * 20)
        for step in result["steps"]:
            logger().info(f"Step {step['step_number']}: {step['total_cost_formatted']}")
        logger().info("")

        logger().info("METHODOLOGY:")
        logger().info("-" * 15)
        for line in result["methodology"].strip().split("\n"):
            if line.strip():
                logger().info(line.strip())
        logger().info("")

        logger().info("LIMITATIONS:")
        logger().info("-" * 15)
        for line in result["limitations"].strip().split("\n"):
            if line.strip():
                logger().info(line.strip())
        logger().info("")

        logger().info("MODEL PRICING USED:")
        logger().info("-" * 20)
        for model, costs in self.model_costs.items():
            if model != "default":  # Don't show default unless it was actually used
                logger().info(f"{model}:")
                logger().info(f"  Input: ${costs['input']*1000:.4f} per 1K tokens")
                logger().info(f"  Output: ${costs['output']*1000:.4f} per 1K tokens")

        logger().info("=" * 50)
