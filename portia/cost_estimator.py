from portia.plan import Plan
from portia.logger import logger
from portia.config import Config
from portia.model import GenerativeModel
from pydantic import BaseModel, Field

class TokenCount(BaseModel):
    """A structured representation of token counts."""
    input_tokens: int = Field(..., description="The number of tokens in the input prompt.")
    output_tokens: int = Field(..., description="The number of tokens in the generated response.")

# Cost per 1K tokens (USD) for Gemini models
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
}


class CostEstimator:
    def __init__(self, portia=None, estimator_model: str = "google/gemini-2.0-flash-lite"):
        self.model_costs = MODEL_PRICING
        self.estimator_model = estimator_model

        config = portia.config if portia else Config.from_default()
        # ✅ This will return a fully configured GenerativeModel
        self.estimator: GenerativeModel = config.get_generative_model(self.estimator_model)

    def plan_estimate(self, plan: Plan) -> dict:
        logger().info("Estimating cost for plan via LLM calls")
        step_costs = []
        total_cost = 0.0

        for step in plan.steps:
            step_cost, details = self._estimate_step_cost_llm(step)
            step_costs.append({
                "step": str(step),
                "cost": step_cost,
                "details": details,
            })
            total_cost += step_cost

        result = {"total_cost": total_cost, "steps": step_costs}
        logger().info(f"Estimated total cost: ${total_cost:.7f}")
        return result

    def _estimate_step_cost_llm(self, step) -> tuple[float, dict]:
        model_name = getattr(step, "model", self.estimator_model)
        pricing = self.model_costs.get(model_name)
        if not pricing:
            logger().warning(f"No pricing found for model {model_name}, skipping cost estimate.")
            return 0.0, {}

        prompt = f"""
You are a cost estimator for LLM steps.
Given the following step in an execution plan, estimate how many input tokens and output tokens 
it is likely to use. Provide integers only.

Step description: {getattr(step, 'description', str(step))}
Task: {getattr(step, 'task', '')}
Inputs: {getattr(step, 'inputs', {})}
Tool ID: {getattr(step, 'tool_id', 'llm_tool')}
Output variable: {getattr(step, 'output', '')}

Return JSON with keys: input_tokens, output_tokens.
"""

        from portia.model import Message
        response: TokenCount = self.estimator.get_structured_response(
            messages=[Message(role="user", content=prompt)],
            schema=TokenCount 
        )
        print(response)

        input_tokens = response.input_tokens
        output_tokens = response.output_tokens

        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
        total_cost = input_cost + output_cost

        details = {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
        }
        return total_cost, details
