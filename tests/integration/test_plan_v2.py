"""Integration tests for PlanV2 examples."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, Field

from portia import Config, LogLevel, Portia
from portia.builder.plan_builder_v2 import PlanBuilderError, PlanBuilderV2
from portia.builder.reference import Input, StepOutput
from portia.clarification_handler import ClarificationHandler
from portia.config import StorageClass
from portia.execution_hooks import ExecutionHooks
from portia.plan_run import PlanRunState
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from portia.clarification import (
        Clarification,
        InputClarification,
        MultipleChoiceClarification,
        UserVerificationClarification,
    )


class CommodityPrice(BaseModel):
    """Price of a commodity."""

    price: float


class CommodityPriceWithCurrency(BaseModel):
    """Price of a commodity with currency."""

    price: float
    currency: str


class GoldPurchaseFinalOutput(BaseModel):
    """Final output of the gold purchase plan."""

    receipt: str


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
def test_simple_builder(is_async: bool) -> None:
    """Test the example from example_builder.py."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
    )

    portia = Portia(config=config)

    plan = (
        PlanBuilderV2("Calculate gold purchase cost and write a poem")
        .input(name="purchase_quantity", description="The quantity of gold to purchase in ounces")
        .invoke_tool_step(
            step_name="Search gold price",
            tool="search_tool",
            args={
                "search_query": "What is the price of gold per ounce in USD?",
            },
            output_schema=CommodityPriceWithCurrency,
        )
        .function_step(
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
        .single_tool_agent_step(
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


def test_plan_v2_conditionals() -> None:
    """Test PlanV2 Conditionals."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        .function_step(
            function=lambda: record_func("if_[1]"),
        )
        .else_if_(
            condition=lambda: True,
            args={},
        )
        .function_step(
            function=lambda: record_func("else_if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["if_[0]", "if_[1]", "final step"]


def test_plan_v2_conditionals_else_if() -> None:
    """Test PlanV2 Conditionals."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: False)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        .function_step(
            function=lambda: record_func("if_[1]"),
        )
        .else_if_(
            condition=lambda: True,
            args={},
        )
        .function_step(
            function=lambda: record_func("else_if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["else_if_[0]", "final step"]


def test_plan_v2_conditionals_else() -> None:
    """Test PlanV2 Conditionals - Else branch."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: False)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        .function_step(
            function=lambda: record_func("if_[1]"),
        )
        .else_if_(
            condition=lambda: False,
            args={},
        )
        .function_step(
            function=lambda: record_func("else_if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["else_[0]", "final step"]


def test_plan_v2_conditionals_nested_branches() -> None:
    """Test PlanV2 Conditionals - Else branch."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        # Start nested branch
        .if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_.if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("if_.else_[0]"),
        )
        .endif()
        # End nested branch
        .else_if_(
            condition=lambda: True,
            args={},
        )
        .function_step(
            function=lambda: record_func("else_if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["if_[0]", "if_.if_[0]", "final step"]


def test_plan_v2_conditionals_nested_branches_else_if() -> None:
    """Test PlanV2 Conditionals - Nested branches - Else if."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        # Start nested branch
        .if_(condition=lambda: False)
        .function_step(
            function=lambda: record_func("if_.if_[0]"),
        )
        .else_if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_.else_if_[0]"),
        )
        .else_if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_.else_if_2[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("if_.else_[0]"),
        )
        .endif()
        # End nested branch
        .else_if_(
            condition=lambda: True,
            args={},
        )
        .function_step(
            function=lambda: record_func("else_if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["if_[0]", "if_.else_if_[0]", "final step"]


def test_plan_v2_unclosed_conditionals() -> None:
    """Test that an unclosed conditional branch in a PlanBuilder raises an error."""
    with pytest.raises(PlanBuilderError):
        (
            PlanBuilderV2(label="Evaluate arbitrary conditionals")
            .if_(condition=lambda: True)
            .function_step(
                function=lambda: None,
            )
            .build()
        )


def test_plan_v2_unclosed_conditionals_complex() -> None:
    """Test that an unclosed conditional branch in a PlanBuilder raises an error."""
    with pytest.raises(PlanBuilderError):
        (
            PlanBuilderV2(label="Evaluate arbitrary conditionals")
            .if_(condition=lambda: True)
            .function_step(
                function=lambda: None,
            )
            # Start nested branch
            .if_(condition=lambda: False)
            .function_step(
                function=lambda: None,
            )
            .else_if_(condition=lambda: True)
            .function_step(
                function=lambda: None,
            )
            .else_if_(condition=lambda: True)
            .function_step(
                function=lambda: None,
            )
            .else_()
            .function_step(
                function=lambda: None,
            )
            # End nested branch
            .else_if_(
                condition=lambda: True,
                args={},
            )
            .function_step(
                function=lambda: None,
            )
            .else_()
            .function_step(
                function=lambda: None,
            )
            .endif()
            .function_step(
                function=lambda: None,
            )
            .build()
        )


def test_plan_v2_conditional_if_without_else_if() -> None:
    """Test else_if is optional."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: False)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        .function_step(
            function=lambda: record_func("if_[1]"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["else_[0]", "final step"]


def test_plan_v2_conditional_if_without_else() -> None:
    """Test else is optional."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: False)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        .function_step(
            function=lambda: record_func("if_[1]"),
        )
        .else_if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("else_if_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["else_if_[0]", "final step"]


def test_plan_v2_conditional_if_without_else_if_or_else() -> None:
    """Test else_if and else are optional."""
    config = Config.from_default(storage_class=StorageClass.CLOUD)
    portia = Portia(config=config)
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=lambda: False)
        .function_step(
            function=lambda: record_func("if_[0]"),
        )
        .function_step(
            function=lambda: record_func("if_[1]"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("final step"),
        )
        .build()
    )
    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert messages == ["final step"]


def test_plan_v2_legacy_condition_string() -> None:
    """Test PlanV2 Legacy Condition String."""

    def dummy(message: str) -> None:
        pass

    def evals_true() -> bool:
        return True

    plan = (
        PlanBuilderV2(label="Evaluate arbitrary conditionals")
        .if_(condition=evals_true)  # None
        .function_step(
            function=lambda: dummy("if_[0]"),
        )
        # Start nested branch
        .if_(condition=evals_true)
        .function_step(
            function=lambda: dummy("if_.if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: dummy("if_.else_[0]"),
        )
        .endif()
        # End nested branch
        .else_if_(
            condition=evals_true,
            args={},
        )
        .function_step(
            function=lambda: dummy("else_if_[0]"),
        )
        .else_()
        .function_step(
            function=lambda: dummy("else_[0]"),
        )
        .endif()
        .function_step(
            function=lambda: dummy("final step"),
        )
        .build()
    )
    condition_strings = [s.to_legacy_step(plan).condition for s in plan.steps]
    assert condition_strings == [
        None,  # 0: initial if_ conditional
        "If $step_0_output is true",
        "If $step_0_output is true",  # 2: nested if_ conditional
        "If $step_2_output is true and $step_0_output is true",
        "If $step_2_output is false and $step_0_output is true",  # 4: nested else_ step
        "If $step_4_output is true and $step_2_output is false and $step_0_output is true",
        "If $step_0_output is true",  # 6: nested endif
        "If $step_0_output is false",  # 7: initial else_if_ conditional
        "If $step_7_output is true and $step_0_output is false",
        "If $step_0_output is false and $step_7_output is false",  # 9: initial else_ conditional
        "If $step_9_output is true and $step_0_output is false and $step_7_output is false",
        None,  # 11: final endif
        None,  # 12: final step
    ]


@pytest.mark.parametrize(("input_value", "expected_output"), [(4, 3), (6, 7)])
def test_conditional_plan_with_record_functions(input_value: int, expected_output: int) -> None:
    """Test a plan with conditional logic that calls different record functions."""

    class ConditionalTestOutput(BaseModel):
        """Output schema for conditional test."""

        value_being_processed: int = Field(description="The value being processed")

    portia = Portia(config=Config.from_default(default_log_level=LogLevel.DEBUG))

    # Track which functions were called
    called_functions: list[str] = []

    def record_high_value() -> int:
        """Record that high value branch was taken and return incremented value."""
        called_functions.append("high_value_branch")
        return input_value + 1

    def record_low_value() -> int:
        """Record that low value branch was taken and return decremented value."""
        called_functions.append("low_value_branch")
        return input_value - 1

    plan = (
        PlanBuilderV2()
        .input(name="number_input", description="An integer to process")
        .if_(
            condition=lambda number_input: number_input > 5,
            args={"number_input": Input("number_input")},
        )
        .function_step(
            function=record_high_value,
            step_name="record_high",
        )
        .else_()
        .function_step(
            function=record_low_value,
            step_name="record_low",
        )
        .endif()
        .llm_step(
            task="Generate a message about the chosen value being processed",
            inputs=[StepOutput("record_high"), StepOutput(5)],
            step_name="print_message",
        )
        .final_output(
            output_schema=ConditionalTestOutput,
        )
        .build()
    )

    plan_run = portia.run_plan(plan, plan_run_inputs={"number_input": input_value})

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None

    final_output = plan_run.outputs.final_output.get_value()
    assert isinstance(final_output, ConditionalTestOutput)
    assert final_output.value_being_processed == expected_output


class TestClarificationHandler(ClarificationHandler):
    """Test clarification handler that automatically selects the laser-sharks-ballad poem."""

    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle multiple choice clarification by selecting the laser-sharks-ballad poem."""
        if clarification.argument_name == "filename" and "poem.txt" in str(clarification.options):
            # Select the first option that contains laser-sharks
            for option in clarification.options:
                if "laser-sharks" in option:
                    on_resolution(clarification, option)
                    return
        raise RuntimeError("Received unexpected clarification")


def test_plan_v2_with_tool_clarification() -> None:
    """Test PlanV2 with a tool that requires clarification handling."""
    portia = Portia(
        config=Config.from_default(
            default_log_level=LogLevel.DEBUG,
        ),
        execution_hooks=ExecutionHooks(clarification_handler=TestClarificationHandler()),
    )

    plan = (
        PlanBuilderV2()
        # This step will throw a clarification as there are two poem.txt in the repo
        .single_tool_agent_step(
            tool="file_reader_tool",
            task="Read the poem in poem.txt from file.",
            step_name="read_poem",
        )
        .llm_step(
            task="Write a review of the poem",
            inputs=[StepOutput("read_poem")],
            step_name="review_poem",
        )
        .build()
    )

    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None

    # Verify that the poem was read successfully
    read_poem_output = plan_run.outputs.step_outputs["$step_0_output"]
    assert read_poem_output is not None
    poem_content = read_poem_output.get_value()
    assert isinstance(poem_content, str)
    assert len(poem_content) > 0
    assert "laser-shark" in poem_content.lower()

    # Verify that a review was written
    review_output = plan_run.outputs.step_outputs["$step_1_output"]
    assert review_output is not None
    review_content = review_output.get_value()
    assert isinstance(review_content, str)
    assert len(review_content) > 0


class ExampleBuilderClarificationHandler(ClarificationHandler):
    """Test clarification handler for the example_builder.py plan."""

    def handle_multiple_choice_clarification(
        self,
        clarification: MultipleChoiceClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle multiple choice clarification by selecting 100."""
        if "How many ounces of gold" in clarification.user_guidance:
            on_resolution(clarification, 100)
            return
        raise RuntimeError("Received unexpected multiple choice clarification")

    def handle_input_clarification(
        self,
        clarification: InputClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle input clarification by returning '100' as string."""
        if "How many ounces of gold" in clarification.user_guidance:
            on_resolution(clarification, "100")
            return
        raise RuntimeError("Received unexpected input clarification")

    def handle_user_verification_clarification(
        self,
        clarification: UserVerificationClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle user verification clarification by accepting."""
        if "Do you want to proceed with the purchase" in clarification.user_guidance:
            on_resolution(clarification, True)  # noqa: FBT003
            return
        raise RuntimeError("Received unexpected user verification clarification")


@pytest.mark.parametrize(
    (
        "storage_class",
        "user_input_options",
        "currency_override",
        "expected_currency",
        "expected_user_input",
    ),
    [
        # Test disk storage class with multi-choice clarification
        (StorageClass.DISK, [50, 100, 200], "USD", "USD", 100),
        # Test input clarification
        (StorageClass.CLOUD, None, "USD", "USD", "100"),
        # Test with default currency (GBP)
        (StorageClass.CLOUD, [50, 100, 200], None, "GBP", 100),
    ],
)
def test_example_builder_plan_scenarios(
    storage_class: StorageClass,
    user_input_options: list[int] | None,
    currency_override: str | None,
    expected_currency: str,
    expected_user_input: int | str,
) -> None:
    """Test the gold purchase plan from example_builder.py with different scenarios."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        storage_class=storage_class,
    )

    portia = Portia(
        config=config,
        execution_hooks=ExecutionHooks(clarification_handler=ExampleBuilderClarificationHandler()),
    )

    def calculate_total_price(
        price_with_currency: CommodityPriceWithCurrency, purchase_quantity: str | int
    ) -> float:
        """Calculate total price with string to int conversion."""
        return price_with_currency.price * int(purchase_quantity)

    plan = (
        PlanBuilderV2("Buy some gold")
        .input(
            name="currency",
            description="The currency to purchase the gold in",
            default_value="GBP",
        )
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
            options=user_input_options,
        )
        .function_step(
            step_name="Calculate total price",
            function=calculate_total_price,
            args={
                "price_with_currency": StepOutput("Search gold price"),
                "purchase_quantity": StepOutput(1),
            },
        )
        .user_verify(
            message=(
                f"Do you want to proceed with the purchase? Price is "
                f"{StepOutput('Calculate total price')}"
            )
        )
        .if_(
            condition=lambda total_price: total_price > 100,
            args={"total_price": StepOutput("Calculate total price")},
        )
        .function_step(function=lambda: "Hey big spender!")
        .else_()
        .function_step(function=lambda: "We need more gold!")
        .endif()
        .llm_step(
            task="Create a fake receipt for the purchase of gold.",
            inputs=[StepOutput("Calculate total price"), Input("currency")],
            step_name="Generate receipt",
        )
        .final_output(
            output_schema=GoldPurchaseFinalOutput,
        )
        .build()
    )

    plan_run_inputs = {}
    if currency_override is not None:
        plan_run_inputs["currency"] = currency_override

    plan_run = portia.run_plan(plan, plan_run_inputs=plan_run_inputs)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None

    # Verify the gold price was found with correct currency
    gold_price_output = plan_run.outputs.step_outputs["$step_0_output"]
    assert gold_price_output is not None
    gold_price_data = gold_price_output.get_value()
    assert isinstance(gold_price_data, CommodityPriceWithCurrency)
    assert gold_price_data.price > 0
    assert gold_price_data.currency == expected_currency

    # Verify user input was correct type and value
    user_input_output = plan_run.outputs.step_outputs["$step_1_output"]
    assert user_input_output is not None
    assert user_input_output.get_value() == expected_user_input

    # Verify total price calculation
    total_price_output = plan_run.outputs.step_outputs["$step_2_output"]
    assert total_price_output is not None
    total_price = total_price_output.get_value()
    assert total_price == gold_price_data.price * 100

    # Verify final output structure
    final_output = plan_run.outputs.final_output.get_value()
    assert isinstance(final_output, GoldPurchaseFinalOutput)
    assert isinstance(final_output.receipt, str)
    assert len(final_output.receipt) > 0
