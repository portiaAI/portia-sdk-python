"""A simple example of using the PlanBuilderV2.

This example demonstrates how to build and execute a multi-step agentic workflow
that combines tool calling, user interactions, conditional logic, and LLM tasks.
"""

from dotenv import load_dotenv
from pydantic import BaseModel

from portia import Input, PlanBuilderV2, Portia, StepOutput
from portia.cli import CLIExecutionHooks

load_dotenv()


# Define output schemas using Pydantic - these structure the data returned by steps
class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity."""

    price: float
    currency: str


class FinalOutput(BaseModel):
    """Final output of the plan."""

    receipt: str
    email_address: str


# Initialize Portia with CLI hooks for interactive prompts and progress display
portia = Portia(execution_hooks=CLIExecutionHooks())

# Build a multi-step plan using our plan builder
plan = (
    PlanBuilderV2("Buy some gold")  # Plan description
    # Define plan inputs - Input() references these throughout the plan
    .input(name="currency", description="The currency to purchase the gold in", default_value="GBP")
    # Call a search tool to get current gold price
    .invoke_tool_step(
        step_name="Search gold price",
        tool="search_tool",
        args={
            "search_query": f"What is the price of gold per ounce in {Input('currency')}?",
        },
        output_schema=CommodityPriceWithCurrency,  # Structure the tool's output
    )
    # Interactive user input with predefined options to get the purchase quantity
    .user_input(
        message="How many ounces of gold do you want to purchase?",
        options=[50, 100, 200],
    )
    # Pure function step for calculation of the total price
    .function_step(
        step_name="Calculate total price",
        function=lambda price_with_currency, purchase_quantity: (
            price_with_currency.price * purchase_quantity
        ),
        args={
            # StepOutput() references previous step results - can use either step name or number
            "price_with_currency": StepOutput("Search gold price"),
            "purchase_quantity": StepOutput(1),
        },
    )
    # User verification with dynamic message - this will exit the plan if the user rejects
    .user_verify(
        message=(
            f"Do you want to proceed with the purchase? Price is "
            f"{StepOutput('Calculate total price')}"
        )
    )
    # Conditional logic based on calculated price
    .if_(
        condition=lambda total_price: total_price > 100,  # noqa: PLR2004
        args={"total_price": StepOutput("Calculate total price")},
    )
    .function_step(function=lambda: print("Hey big spender!"))  # noqa: T201
    .else_()
    .function_step(function=lambda: print("We need more gold!"))  # noqa: T201
    .endif()
    # LLM step for generating a receipt using natural language instructions
    .llm_step(
        task="Create a fake receipt for the purchase of gold.",
        inputs=[StepOutput("Calculate total price"), Input("currency")],
    )
    # Single tool agent step - LLM uses a specific tool to complete a task
    .single_tool_agent_step(
        task="Send the receipt to Robbie in an email at not_an_email@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput(9)],  # Reference the receipt from previous step
    )
    # Define the final output structure
    .final_output(
        output_schema=FinalOutput,
    )
    .build()  # Build the plan object
)

# Execute the plan with runtime inputs (overrides default currency)
plan_run = portia.run_plan(
    plan,
    plan_run_inputs={"currency": "USD"},
)
print(plan_run)  # noqa: T201
