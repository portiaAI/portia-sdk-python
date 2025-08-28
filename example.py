"""A simple example of using the PlanBuilderV2."""

from dotenv import load_dotenv
from pydantic import BaseModel

from portia import Input, PlanBuilderV2, Portia, StepOutput
from portia.cli import CLIExecutionHooks

load_dotenv(override=True)


class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity."""

    price: float
    currency: str


class FinalOutput(BaseModel):
    """Final output of the plan."""

    receipt: str
    email_address: str


portia = Portia(execution_hooks=CLIExecutionHooks())

mini_plan = (
    PlanBuilderV2("Do a random step")
    .input(
        name="message", description="The message to display to the user", default_value="Skibidi"
    )
    .user_verify(message=f"{Input('message')}")
    .build()
)

plan = (
    PlanBuilderV2("Buy some gold")
    .input(name="currency", description="The currency to purchase the gold in", default_value="GBP")
    .invoke_tool_step(
        step_name="Search gold price",
        tool="search_tool",
        args={
            "search_query": f"What is the price of gold per ounce in {Input('currency')}?",
        },
        output_schema=CommodityPriceWithCurrency,
    )
    .user_input(
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
            "purchase_quantity": StepOutput(1),
        },
    )
    .add_steps(
        mini_plan,
        input_values={
            "message": f"Do you want to proceed with the purchase? Price is "
            f"{StepOutput('Calculate total price')}"
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
        task="Create a fake receipt for the purchase of gold.",
        inputs=[StepOutput("Calculate total price"), Input("currency")],
    )
    .single_tool_agent_step(
        task="Send the receipt to Robbie in an email at not_an_email@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput(9)],
    )
    .final_output(
        output_schema=FinalOutput,
    )
    .build()
)

plan_run = portia.run_plan(
    plan,
    plan_run_inputs={"currency": "USD"},
)
print(plan_run)  # noqa: T201
