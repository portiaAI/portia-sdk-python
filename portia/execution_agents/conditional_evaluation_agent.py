"""Conditional evaluation agent for PlanV2."""

from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.config import Config
from portia.logger import logger
from portia.model import Message


class BooleanResponse(BaseModel):
    """Boolean response for conditional evaluation."""

    explanation: str = Field(description="Explanation for the response value")
    response: bool = Field(description="Whether the conditional statement is True")


class ConditionalEvaluationAgent:
    """Conditional evaluation agent for PlanV2."""

    def __init__(self, config: Config) -> None:
        """Initialize the conditional evaluation agent."""
        self.config = config

    @traceable(name="Conditional Evaluation Agent - Execute")
    async def execute(self, conditional: str, arguments: dict[str, Any]) -> bool:
        """Execute the conditional evaluation agent."""
        model = self.config.get_introspection_model()
        rendered_args = "\n".join(
            f"<arg><name>{k}</name><value>{v}</value></arg>" for k, v in arguments.items()
        )
        resp = await model.aget_structured_response(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are a helpful assistant that evaluates whether conditional "
                        "statements should evaluate to true or false.\nConditional statements "
                        "can be highly complex, so use your reasoning power to ensure accuracy "
                        "of evaluation."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"The conditional statement to evaluate is:\n```{conditional}```\n"
                        f"The following pieces of context will be necessary:\n{rendered_args}\n"
                    ),
                ),
            ],
            schema=BooleanResponse,
        )
        logger().info(f"Conditional evaluation agent response: {resp}")
        return resp.response
