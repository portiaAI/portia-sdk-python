"""A simple example of using the PlanBuilderV2.

This example demonstrates how to build and execute a multi-step agentic workflow
that combines tool calling, user interactions, conditional logic, and LLM tasks.
"""

from dotenv import load_dotenv
from pydantic import BaseModel

from portia import Input, PlanBuilderV2, Portia, StepOutput

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


# Initialize Portia
portia = Portia()

# Build a multi-step plan using our plan builder
plan = (
    PlanBuilderV2("Buy some gold")  # Plan description
    # Define plan inputs - Input() references these throughout the plan
    .input(name="country", description="The country to purchase the gold in", default_value="UK")
    # Call a search tool to get the currency for the provided country
    .invoke_tool_step(
        step_name="Search currency",
        tool="search_tool",
        args={
            # You can reference inputs to the plan using Input()
            "search_query": f"What is the currency in {Input('country')}?",
        },
    )
    # Use an agent (ReAct = Reasoning + Acting) with a search tool and a calculator tool to
    # determine the price of gold in that country (with the correct currency)
    .react_agent_step(
        step_name="Search gold price",
        tools=["search_tool", "calculator_tool"],
        task=f"What is the price of gold per ounce in {Input('country')}?",
        # You can reference past outputs using StepOutput()
        inputs=[StepOutput("Search currency")],
        output_schema=CommodityPriceWithCurrency,  # Structure the agent's output
    )
    # Interactive user input with predefined options to get the purchase quantity
    .user_input(
        step_name="Purchase quantity",
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
            # StepOutput can use either step name or number
            "price_with_currency": StepOutput(1),
            "purchase_quantity": StepOutput("Purchase quantity"),
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
        step_name="Generate receipt",
        task="Create a fake receipt for the purchase of gold.",
        inputs=[StepOutput("Calculate total price"), StepOutput("Purchase quantity")],
    )
    # Single tool agent step - LLM uses a specific tool to complete a task
    .single_tool_agent_step(
        task="Send the receipt to Robbie in an email at not_an_email@portialabs.ai",
        tool="portia:google:gmail:send_email",
        inputs=[StepOutput("Generate receipt")],
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
    plan_run_inputs={"country": "Spain"},
)
print(plan_run)  # noqa: T201

# example for_each loop
plan = (
    PlanBuilderV2("Buy some gold")
    .function_step(function=lambda: [1, 2, 3], step_name="Items")  # create a list of items
    .loop(
        over=StepOutput("Items"), step_name="Loop"
    )  # name the loop, iterate over the list of items
    .function_step(
        function=lambda item: item + 1, step_name="test_function", args={"item": StepOutput("Loop")}
    )  # call the loop step to access current item
    .end_loop(step_name="end_loop")
).build()


# example while loop
class WhileCondition:
    """Class that is used to track the number of times the condition has been run."""

    counter = 0

    def run(self) -> bool:  # noqa: D102
        self.counter += 1
        return self.counter <= 4  # noqa: PLR2004


while_condition = WhileCondition()
plan = (
    PlanBuilderV2("Buy some gold")
    .loop(
        while_=while_condition.run, step_name="Loop"
    )  # name the loop, iterate while the condition is true
    .function_step(
        function=lambda: print("Hello"),  # noqa: T201
        step_name="test_function",
    )  # call the loop step to access current item
    .end_loop(step_name="end_loop")
).build()

# example do_while loop


class Iterator:
    """Class that is used to track the number of times the iterator has been run."""

    counter = 0

    def run(self) -> int:  # noqa: D102
        self.counter += 1
        return self.counter


iterator = Iterator()
plan = (
    PlanBuilderV2("Buy some gold")
    .loop(
        do_while_=lambda x: x < 4,  # noqa: PLR2004
        args={"x": StepOutput("test_function")},
        step_name="Loop",
    )  # can access the item from the loop as a variable as it gets evaluated at end of loop
    .function_step(function=lambda: iterator.run(), step_name="test_function")
    .end_loop(step_name="end_loop")
).build()
