"""Simple Example."""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from portia import Config, LogLevel
from portia.builder.plan_builder import PlanBuilder
from portia.builder.step import StepOutput
from portia.cli import CLIExecutionHooks
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
        return f"{amount.price * 1.2} {currency_to}"


class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity."""

    price: float
    currency: str


class FinalOutput(BaseModel):
    """Final output of the plan."""

    poem: str
    email_address: str


def only_continue_if_affordable(cost: float) -> bool:
    """Only continue if the price is affordable."""
    return cost < 500_000


def always_continue(price: str) -> bool:
    """Always continue."""
    return True


config = Config.from_default(default_log_level=LogLevel.DEBUG)
portia = Portia(
    config=config,
    tools=InMemoryToolRegistry.from_local_tools([CurrencyConversionTool()])
    + DefaultToolRegistry(config=config),
    execution_hooks=CLIExecutionHooks(),
)

purchase_quantity = 100
plan = (
    PlanBuilder("Write a poem about the price of gold")
    .single_tool_agent(
        name="Search gold price",
        tool="search_tool",
        task="Search for the price of gold per ounce in USD",
        output_schema=CommodityPrice,
    )
    .tool_call(
        tool="currency_conversion_tool",
        args={
            "amount": StepOutput("Search gold price"),
            "currency_from": "USD",
            "currency_to": "GBP",
        },
        output_schema=CommodityPriceWithCurrency,
    )
    .function_call(
        function=lambda price_with_currency: purchase_quantity * price_with_currency.price,
        args={"price_with_currency": StepOutput(1)},
    )
    .hook(only_continue_if_affordable, args={"cost": StepOutput(2)})
    .llm_step(
        task="Write a poem about the price of gold",
        inputs=[StepOutput(step=0)],
    )
    .single_tool_agent(
        task="Send the poem to Robbie in an email at robbie+test@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput(step=4)],
    )
    .final_output(
        output_schema=FinalOutput,
    )
    .build()
)

# Test async
result1 = asyncio.run(portia.arun_plan(plan))
print(result1)

# Test sync
# result2 = portia.run_plan(plan)
# print(result2)

# Test clarifications
# result3 = asyncio.run(portia.arun_plan(plan, end_user=EndUser(external_id=str(uuid.uuid4()))))
# print(result3)
