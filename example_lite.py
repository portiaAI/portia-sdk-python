"""Simple Example."""

from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from portia import Config, LogLevel
from portia.portia_lite import PlanBuilderLite, PortiaLite
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import DefaultToolRegistry, InMemoryToolRegistry

load_dotenv()


class CurrencyConversionToolSchema(BaseModel):
    """Schema defining the inputs for the CurrencyConversionTool."""

    amount: str = Field(..., description="The amount to convert")
    currency_from: str = Field(..., description="The currency to convert from")
    currency_to: str = Field(..., description="The currency to convert to")


class CurrencyConversionTool(Tool[str]):
    """Converts currency."""

    id: str = "currency_conversion_tool"
    name: str = "Currency conversion tool"
    description: str = "Converts money between currencies"
    args_schema: type[BaseModel] = CurrencyConversionToolSchema
    output_schema: tuple[str, str] = ("str", "The converted amount")

    def run(self, _: ToolRunContext, amount: str, currency_from: str, currency_to: str) -> str:
        """Run the CurrencyConversionTool."""
        if isinstance(amount, str) and amount.startswith("price="):
            amount_value = amount.split("=", 1)[1]
        else:
            amount_value = amount
        return f"{float(amount_value) * 1.2}"


class CommodityPrice(BaseModel):
    """Price of a commodity."""

    price: float


def only_continue_if_affordable(price: str) -> bool:
    """Only continue if the price is affordable."""
    return float(price) < 5000


def always_continue(outputs: list[Any]) -> bool:
    """Always continue."""
    return True


config = Config.from_default(default_log_level=LogLevel.DEBUG)
portia = PortiaLite(
    config=config,
    tools=InMemoryToolRegistry.from_local_tools([CurrencyConversionTool()])
    + DefaultToolRegistry(config=config),
)


plan = (
    PlanBuilderLite()
    .agent_step(
        tool="search_tool",
        query="Search for the price of gold in USD",
        output_class=CommodityPrice,
        condition=always_continue,
    )
    .tool_step(
        tool="currency_conversion_tool",
        inputs={"amount": "$output0", "currency_from": "USD", "currency_to": "GBP"},
    )
    .hook(only_continue_if_affordable, inputs={"price": "$output1"})
    .llm_step(
        query="The price of gold is $output1. Write a poem about the price of gold",
    )
    .agent_step(
        tool="portia:google:gmail:send_email",
        query="A poem about the price of gold is $output2. Send this poem to Robbie in an email at "
        "robbi+test@portialabs.ai",
        condition="Continue as long as the email doesn't have any swear words",
    )
    .build()
)

portia.run(plan)
