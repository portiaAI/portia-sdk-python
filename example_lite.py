"""Simple Example."""

import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from portia import Config, LogLevel
from portia.builder.plan_builder import PlanBuilder
from portia.builder.step import StepOutput
from portia.portia import Portia
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import DefaultToolRegistry, InMemoryToolRegistry

load_dotenv()


class CommodityPrice(BaseModel):
    """Price of a commodity."""

    price: float


class CurrencyConversionToolSchema(BaseModel):
    """Schema defining the inputs for the CurrencyConversionTool."""

    amount: CommodityPrice = Field(..., description="The amount to convert")
    currency_from: str = Field(..., description="The currency to convert from")
    currency_to: str = Field(..., description="The currency to convert to")


class CurrencyConversionTool(Tool[str]):
    """Converts currency."""

    id: str = "currency_conversion_tool"
    name: str = "Currency conversion tool"
    description: str = "Converts money between currencies"
    args_schema: type[BaseModel] = CurrencyConversionToolSchema
    output_schema: tuple[str, str] = ("str", "The converted amount")

    def run(
        self, _: ToolRunContext, amount: CommodityPrice, currency_from: str, currency_to: str
    ) -> str:
        """Run the CurrencyConversionTool."""
        return f"{amount.price * 1.2}"


def only_continue_if_affordable(price: str) -> bool:
    """Only continue if the price is affordable."""
    return float(price) < 5000


def always_continue(price: str) -> bool:
    """Always continue."""
    return True


config = Config.from_default(default_log_level=LogLevel.DEBUG)
portia = Portia(
    config=config,
    tools=InMemoryToolRegistry.from_local_tools([CurrencyConversionTool()])
    + DefaultToolRegistry(config=config),
)

plan = (
    PlanBuilder()
    .single_tool_agent(
        name="Search gold price",
        tool="search_tool",
        task="Search for the price of gold in USD",
        output_schema=CommodityPrice,
    )
    .tool_call(
        tool="currency_conversion_tool",
        args={
            "amount": StepOutput("Search gold price"),
            "currency_from": "USD",
            "currency_to": "GBP",
        },
    )
    .hook(only_continue_if_affordable, args={"price": StepOutput(1)})
    .llm_step(
        task="Write a poem about the price of gold",
        inputs=[StepOutput(step=0)],
    )
    .single_tool_agent(
        task="Send the poem to Robbie in an email at robbie+test@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput(step=3)],
    )
    .build()
)

result = asyncio.run(portia.arun_plan(plan))
