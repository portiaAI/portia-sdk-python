"""A simple example of using the PlanBuilderV2."""

from dotenv import load_dotenv
from pydantic import BaseModel

from portia import Input, PlanBuilderV2, Portia, StepOutput
from portia.cli import CLIExecutionHooks

load_dotenv()


class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity."""

    price: float
    currency: str


class FinalOutput(BaseModel):
    """Final output of the plan."""

    receipt: str
    email_address: str


portia = Portia(execution_hooks=CLIExecutionHooks())

plan = (
    PlanBuilderV2("Buy some gold")
    .input(name="country", description="The country to purchase the gold in", default_value="UK")
    .invoke_tool_step(
        step_name="Search currency",
        tool="search_tool",
        args={
            "search_query": f"What is the currency in {Input('country')}?",
        },
    )
    .react_agent_step(
        step_name="Search gold price",
        tools=["search_tool", "calculator_tool"],
        task=f"What is the price of gold per ounce in {Input('country')}?",
        inputs=[StepOutput(0)],
        output_schema=CommodityPriceWithCurrency,
    )
    .user_input(
        step_name="Purchase quantity",
        message="How many ounces of gold do you want to purchase?",
        options=[50, 100, 200],
    )
    .function_step(
        step_name="Calculate total price",
        function=lambda price_with_currency, purchase_quantity: (
            price_with_currency.price * purchase_quantity
        ),
        args={
            "price_with_currency": StepOutput("Search gold price"),
            "purchase_quantity": StepOutput("Purchase quantity"),
        },
    )
    .user_verify(
        message=(
            f"Do you want to proceed with the purchase? Price is "
            f"{StepOutput('Calculate total price')}"
        )
    )
    .if_(
        condition=lambda total_price: total_price > 100,  # noqa: PLR2004
        args={"total_price": StepOutput("Calculate total price")},
    )
    .function_step(function=lambda: print("Hey big spender!"))  # noqa: T201
    .else_()
    .function_step(function=lambda: print("We need more gold!"))  # noqa: T201
    .endif()
    .llm_step(
        step_name="Generate receipt",
        task="Create a fake receipt for the purchase of gold.",
        inputs=[StepOutput("Calculate total price"), StepOutput("Purchase quantity")],
    )
    .single_tool_agent_step(
        task="Send the receipt to Robbie in an email at not_an_email@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput("Generate receipt")],
    )
    .final_output(
        output_schema=FinalOutput,
    )
    .build()
)

plan_run = portia.run_plan(
    plan,
    plan_run_inputs={"country": "Spain"},
)
print(plan_run)  # noqa: T201
