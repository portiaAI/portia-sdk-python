"""Integration tests for PlanV2 examples."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from portia import Config, LogLevel
from portia.builder.plan_builder import PlanBuilderV2
from portia.builder.reference import Input, StepOutput
from portia.plan_run import PlanRunState
from portia.portia import Portia
from portia.tool import Tool, ToolRunContext


class CommodityPrice(BaseModel):
    """Price of a commodity."""

    price: float


class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity with currency."""

    price: float
    currency: str


class CurrencyConversionResult(BaseModel):
    """Result of currency conversion."""

    converted_amount: str


class CurrencyConversionToolSchema(BaseModel):
    """Schema defining the inputs for the CurrencyConversionTool."""

    amount: CommodityPrice = Field(..., description="The amount to convert")
    currency_from: str = Field(..., description="The currency to convert from")
    currency_to: str = Field(..., description="The currency to convert to")


class CurrencyConversionTool(Tool[CurrencyConversionResult]):
    """Converts currency."""

    id: str = "currency_conversion_tool"
    name: str = "Currency conversion tool"
    description: str = "Converts money between currencies"
    args_schema: type[BaseModel] = CurrencyConversionToolSchema
    output_schema: tuple[str, str] = ("CurrencyConversionResult", "The converted amount")

    def run(
        self,
        _: ToolRunContext,
        amount: CommodityPrice,
        currency_from: str,  # noqa: ARG002
        currency_to: str,
    ) -> CurrencyConversionResult:
        """Run the CurrencyConversionTool."""
        converted_amount = f"{amount.price * 1.2} {currency_to}"
        return CurrencyConversionResult(converted_amount=converted_amount)


class FinalOutput(BaseModel):
    """Final output of the plan."""

    poem: str
    example_similar_poem: str


@pytest.mark.parametrize("is_async", [False, True])
def test_example_builder(is_async: bool) -> None:
    """Test the example from example_builder.py."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
    )

    portia = Portia(config=config)

    plan = (
        PlanBuilderV2("Calculate gold purchase cost and write a poem")
        .input(name="purchase_quantity", description="The quantity of gold to purchase in ounces")
        .tool_run(
            step_name="Search gold price",
            tool="search_tool",
            args={
                "search_query": "What is the price of gold per ounce in USD?",
            },
            output_schema=CommodityPriceWithCurrency,
        )
        .function_call(
            function=lambda price_with_currency, purchase_quantity: (
                price_with_currency.price * purchase_quantity
            ),
            args={
                "price_with_currency": StepOutput("Search gold price"),
                "purchase_quantity": Input("purchase_quantity"),
            },
        )
        .llm_step(
            task="Write a poem about the current price of gold in USD",
            inputs=[StepOutput(0)],
        )
        .single_tool_agent(
            task="Search for similar poems about gold",
            tool="search_tool",
            inputs=[StepOutput(2)],
        )
        .final_output(
            output_schema=FinalOutput,
        )
        .build()
    )

    if is_async:
        plan_run = asyncio.run(portia.arun_plan(plan, plan_run_inputs={"purchase_quantity": 100}))
    else:
        plan_run = portia.run_plan(plan, plan_run_inputs={"purchase_quantity": 100})

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None

    final_output = plan_run.outputs.final_output.get_value()
    assert isinstance(final_output, FinalOutput)
    assert isinstance(final_output.poem, str)
    assert len(final_output.poem) > 0
    assert isinstance(final_output.example_similar_poem, str)
    assert len(final_output.example_similar_poem) > 0
