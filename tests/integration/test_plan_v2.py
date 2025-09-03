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


@pytest.fixture
def local_portia() -> Portia:
    """Create a local Portia instance."""
    return Portia(
        config=Config.from_default(
            storage_class=StorageClass.MEMORY, default_log_level=LogLevel.DEBUG, portia_api_key=None
        )
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_simple_builder(is_async: bool) -> None:
    """Test the example from example_builder.py."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
    )

    portia = Portia(config=config)

    plan = (
        PlanBuilderV2("Calculate gold purchase cost and write a poem")
        .input(name="purchase_quantity", description="The quantity of gold to purchase in kilos")
        .invoke_tool_step(
            step_name="Search gold price",
            tool="search_tool",
            args={
                "search_query": "What is the price of gold per kilo in USD?",
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


def test_plan_v2_else_if_before_if_raises_error() -> None:
    """Test that using else_if before if raises a PlanBuilderError."""
    with pytest.raises(PlanBuilderError):
        (
            PlanBuilderV2(label="Invalid conditional order")
            .else_if_(condition=lambda: True)
            .function_step(function=lambda: None)
            .build()
        )


def test_plan_v2_else_before_if_raises_error() -> None:
    """Test that using else before if raises a PlanBuilderError."""
    with pytest.raises(PlanBuilderError):
        (
            PlanBuilderV2(label="Invalid conditional order")
            .else_()
            .function_step(function=lambda: None)
            .build()
        )


def test_plan_v2_endif_before_if_raises_error() -> None:
    """Test that using endif before if raises a PlanBuilderError."""
    with pytest.raises(PlanBuilderError):
        (
            PlanBuilderV2(label="Invalid conditional order")
            .endif()
            .function_step(function=lambda: None)
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
        if "How many kilos of gold" in clarification.user_guidance:
            on_resolution(clarification, 2)
            return
        raise RuntimeError("Received unexpected multiple choice clarification")

    def handle_input_clarification(
        self,
        clarification: InputClarification,
        on_resolution: Callable[[Clarification, object], None],
        on_error: Callable[[Clarification, object], None],  # noqa: ARG002
    ) -> None:
        """Handle input clarification by returning '2' as string."""
        if "How many kilos of gold" in clarification.user_guidance:
            on_resolution(clarification, "2")
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


# Rerun as occasionally Tavily doesn't give back the gold price
@pytest.mark.flaky(reruns=3)
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
        (StorageClass.DISK, [1, 2, 5], "USD", "USD", 2),
        # Test input clarification
        (StorageClass.CLOUD, None, "USD", "USD", "2"),
        # Test with default currency (EUR)
        (StorageClass.CLOUD, [1, 2, 5], None, "EUR", 2),
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
            default_value="EUR",
        )
        .invoke_tool_step(
            step_name="Search gold price",
            tool="search_tool",
            args={
                "search_query": f"What is the price of gold per kilo in {Input('currency')}?",
            },
            output_schema=CommodityPriceWithCurrency,
        )
        .user_input(
            message="How many kilos of gold do you want to purchase?",
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
    assert total_price == gold_price_data.price * 2

    # Verify final output structure
    final_output = plan_run.outputs.final_output.get_value()
    assert isinstance(final_output, GoldPurchaseFinalOutput)
    assert isinstance(final_output.receipt, str)
    assert len(final_output.receipt) > 0


def collect_fn(
    top_no_def: str,
    top_with_def: str,
    sub_no_def_1: str,
    sub_no_def_2: str,
    sub_with_def_1: str,
    sub_with_def_2: str,
) -> dict[str, str]:
    """Collect all input values."""
    return {
        "top_input_no_default": top_no_def,
        "top_input_with_default": top_with_def,
        "sub_input_no_default_1": sub_no_def_1,
        "sub_input_no_default_2": sub_no_def_2,
        "sub_input_with_default_1": sub_with_def_1,
        "sub_input_with_default_2": sub_with_def_2,
    }


def test_plan_v2_input_linking_with_add_steps() -> None:
    """Test input linking between top-level plan and sub-plan using add_steps with input_values."""
    config = Config.from_default()
    portia = Portia(config=config)

    # Create sub-plan with 4 inputs (2 with defaults, 2 without)
    sub_plan = (
        PlanBuilderV2("Sub-plan with multiple inputs")
        .input(name="sub_input_no_default_1", description="Sub input 1 without default")
        .input(name="sub_input_no_default_2", description="Sub input 2 without default")
        .input(
            name="sub_input_with_default_1",
            description="Sub input 1 with default",
            default_value="original_default_1",
        )
        .input(
            name="sub_input_with_default_2",
            description="Sub input 2 with default",
            default_value="original_default_2",
        )
        .function_step(
            function=lambda: "sub_plan_executed",
            step_name="sub_execution_step",
        )
        .build()
    )

    # Create top-level plan with 2 inputs (1 with default, 1 without)
    plan = (
        PlanBuilderV2("Top-level plan with input linking")
        .input(name="top_input_no_default", description="Top input without default")
        .input(
            name="top_input_with_default",
            description="Top input with default",
            default_value="top_default_value",
        )
        .function_step(
            function=lambda: 100,
            step_name="first_number",
        )
        .function_step(
            function=lambda: 200,
            step_name="second_number",
        )
        # Add sub-plan steps with input values for 2 of the 4 sub-plan inputs
        .add_steps(
            sub_plan,
            input_values={
                "sub_input_no_default_1": (
                    f"Number 1: {StepOutput('first_number')}. "
                    f"Number 2: {StepOutput('second_number')}"
                ),
                "sub_input_with_default_1": StepOutput("second_number"),
            },
        )
        # Final step that outputs all input values for verification
        .function_step(
            function=collect_fn,
            args={
                "top_no_def": Input("top_input_no_default"),
                "top_with_def": Input("top_input_with_default"),
                "sub_no_def_1": Input("sub_input_no_default_1"),
                "sub_no_def_2": Input("sub_input_no_default_2"),
                "sub_with_def_1": Input("sub_input_with_default_1"),
                "sub_with_def_2": Input("sub_input_with_default_2"),
            },
            step_name="collect_all_inputs",
        )
        .build()
    )

    # Run the plan with only 2 input values:
    # - One for top-level plan without default
    # - One for sub-plan without default that wasn't set via input_values
    plan_run = portia.run_plan(
        plan,
        plan_run_inputs={
            "top_input_no_default": "top_value",
            "sub_input_no_default_2": "sub_value",
        },
    )

    assert plan_run.state == PlanRunState.COMPLETE

    # Verify final step collected all input values correctly
    final_output = plan_run.outputs.step_outputs["$step_3_output"]
    all_inputs = final_output.get_value()

    # Verify all input values are as expected
    assert all_inputs is not None
    assert all_inputs.get("top_input_no_default", "") == "top_value"
    assert all_inputs.get("top_input_with_default") == "top_default_value"
    assert all_inputs.get("sub_input_no_default_1", "") == "Number 1: 100. Number 2: 200"
    assert all_inputs.get("sub_input_no_default_2", "") == "sub_value"
    assert all_inputs.get("sub_input_with_default_1", "") == 200
    assert all_inputs.get("sub_input_with_default_2", "") == "original_default_2"


# Loop integration tests


def test_plan_v2_loop_conditional_simple(local_portia: Portia) -> None:
    """Test PlanV2 conditional loop - simple case that runs once."""
    messages: list[str] = []
    counter = 0

    def record_func(message: str) -> None:
        messages.append(message)

    def increment_and_check() -> bool:
        nonlocal counter
        counter += 1
        record_func(f"loop_iteration_{counter}")
        return counter < 2  # Run twice (counter 1 and 2)

    plan = (
        PlanBuilderV2(label="Test conditional loop")
        .function_step(
            function=lambda: record_func("before_loop"),
        )
        .loop(condition=increment_and_check)
        .function_step(
            function=lambda: record_func("inside_loop"),
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("after_loop"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Verify the loop executed correctly
    assert "before_loop" in messages
    assert "after_loop" in messages
    assert counter == 2

    # Check that the loop body executed multiple times
    loop_executions = [msg for msg in messages if msg == "inside_loop"]
    assert len(loop_executions) == 2

    # Check that condition function was called
    condition_calls = [msg for msg in messages if msg.startswith("loop_iteration_")]
    assert len(condition_calls) == 2
    assert "loop_iteration_1" in condition_calls
    assert "loop_iteration_2" in condition_calls


def test_plan_v2_loop_conditional_false(local_portia: Portia) -> None:
    """Test PlanV2 conditional loop - condition is false."""
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    def always_false() -> bool:
        record_func("condition_evaluated")
        return False

    plan = (
        PlanBuilderV2(label="Test conditional loop that is false")
        .function_step(
            function=lambda: record_func("before_loop"),
        )
        .loop(condition=always_false)
        .function_step(
            function=lambda: record_func("inside_loop"),
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("after_loop"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Should execute: before_loop, condition_evaluated, after_loop
    # Note: conditional loops always run at least once to evaluate the condition
    expected_messages = ["before_loop", "inside_loop", "condition_evaluated", "after_loop"]
    assert messages == expected_messages


def test_plan_v2_loop_for_each_simple(local_portia: Portia) -> None:
    """Test PlanV2 for-each loop with simple list."""
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    def generate_list() -> list[str]:
        return ["apple", "banana", "cherry"]

    plan = (
        PlanBuilderV2(label="Test for-each loop")
        .function_step(
            function=generate_list,
            step_name="generate_items",
        )
        .function_step(
            function=lambda: record_func("before_loop"),
        )
        .loop(over=StepOutput("generate_items"), step_name="Loop")
        .function_step(
            function=lambda item: record_func(f"processing_{item}"),
            args={"item": StepOutput("Loop")},
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("after_loop"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    expected_messages = [
        "before_loop",
        "processing_apple",
        "processing_banana",
        "processing_cherry",
        "after_loop",
    ]
    assert messages == expected_messages


def test_plan_v2_loop_for_each_empty_list(local_portia: Portia) -> None:
    """Test PlanV2 for-each loop with empty list."""
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    def generate_empty_list() -> list[str]:
        return []

    plan = (
        PlanBuilderV2(label="Test for-each loop with empty list")
        .function_step(
            function=generate_empty_list,
            step_name="generate_items",
        )
        .function_step(
            function=lambda: record_func("before_loop"),
        )
        .loop(over=StepOutput("generate_items"), step_name="Loop")
        .function_step(
            function=lambda item: record_func(f"processing_{item}"),
            args={"item": StepOutput("Loop")},
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("after_loop"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Should execute: before_loop, after_loop (no loop iterations)
    expected_messages = ["before_loop", "after_loop"]
    assert messages == expected_messages


def test_plan_v2_loop_nested_conditional(local_portia: Portia) -> None:
    """Test PlanV2 nested conditional loops."""
    messages: list[str] = []
    outer_counter = 0
    inner_counter = 0

    def record_func(message: str) -> None:
        messages.append(message)

    def outer_condition() -> bool:
        nonlocal outer_counter
        outer_counter += 1
        record_func(f"outer_condition_{outer_counter}")
        return outer_counter < 3  # Run twice

    def inner_condition() -> bool:
        nonlocal inner_counter
        inner_counter += 1
        record_func(f"inner_condition_{inner_counter}")
        return inner_counter < 2  # Run once per outer iteration

    plan = (
        PlanBuilderV2(label="Test nested conditional loops")
        .function_step(
            function=lambda: record_func("start"),
        )
        .loop(condition=outer_condition)
        .function_step(
            function=lambda: record_func("outer_loop_start"),
        )
        .loop(condition=inner_condition)
        .function_step(
            function=lambda: record_func("inner_loop"),
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("outer_loop_end"),
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("end"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Verify the nested loops executed correctly
    assert "start" in messages
    assert "end" in messages

    # Check that outer loop executed multiple times
    outer_starts = [msg for msg in messages if msg == "outer_loop_start"]
    assert len(outer_starts) == 3  # outer_counter < 3 means it runs 3 times (1, 2, 3)

    # Check that inner loop executed multiple times
    inner_loops = [msg for msg in messages if msg == "inner_loop"]
    assert (
        len(inner_loops) == 4
    )  # inner loop runs multiple times per outer iteration due to condition evaluation

    # Check that condition functions were called
    outer_conditions = [msg for msg in messages if msg.startswith("outer_condition_")]
    assert len(outer_conditions) == 3
    assert "outer_condition_1" in outer_conditions
    assert "outer_condition_2" in outer_conditions

    inner_conditions = [msg for msg in messages if msg.startswith("inner_condition_")]
    assert len(inner_conditions) == 4
    assert "inner_condition_1" in inner_conditions
    assert "inner_condition_2" in inner_conditions

    assert outer_counter == 3
    assert inner_counter == 4


def test_plan_v2_loop_inside_conditional(local_portia: Portia) -> None:
    """Test PlanV2 loop inside conditional block."""
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    plan = (
        PlanBuilderV2(label="Test loop inside conditional")
        .function_step(
            function=lambda: record_func("start"),
        )
        .if_(condition=lambda: True)
        .function_step(
            function=lambda: record_func("if_start"),
        )
        .function_step(
            function=lambda: list(range(3)),
            step_name="generate_items",
        )
        .loop(over=StepOutput("generate_items"), step_name="Loop")
        .function_step(
            function=lambda item: record_func(f"loop_item_{item}"),
            args={"item": StepOutput("Loop")},
            step_name="loop_item",
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("if_end"),
        )
        .else_()
        .function_step(
            function=lambda: record_func("else_block"),
        )
        .endif()
        .function_step(
            function=lambda: record_func("end"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Should execute: start, if_start, loop_item_1, loop_item_2, loop_item_3, if_end, end
    expected_messages = [
        "start",
        "if_start",
        "loop_item_0",
        "loop_item_1",
        "loop_item_2",
        "if_end",
        "end",
    ]
    assert messages == expected_messages


def test_plan_v2_loop_with_args(local_portia: Portia) -> None:
    """Test PlanV2 conditional loop with arguments."""
    messages: list[str] = []
    counter = 0

    def record_func(message: str) -> None:
        messages.append(message)

    def condition_with_args(x: int, y: str) -> bool:
        nonlocal counter
        counter += 1
        record_func(f"condition_eval_{counter}_x_{x}_y_{y}")
        return counter < 3  # Run twice

    plan = (
        PlanBuilderV2(label="Test conditional loop with args")
        .function_step(
            function=lambda: record_func("start"),
        )
        .loop(condition=condition_with_args, args={"x": 42, "y": "test"})
        .function_step(
            function=lambda: record_func("inside_loop"),
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("end"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    expected_messages = [
        "start",
        "inside_loop",
        "condition_eval_1_x_42_y_test",
        "inside_loop",
        "condition_eval_2_x_42_y_test",
        "inside_loop",
        "condition_eval_3_x_42_y_test",
        "end",
    ]
    assert messages == expected_messages
    assert counter == 3


def test_plan_v2_loop_string_condition(local_portia: Portia) -> None:
    """Test PlanV2 conditional loop with string condition."""
    messages: list[str] = []
    counter = 0

    def record_func(message: str) -> None:
        messages.append(message)

    def increment_counter(message: str) -> int:
        nonlocal counter
        counter += 1
        record_func(message)
        return counter

    plan = (
        PlanBuilderV2(label="Test conditional loop with string condition")
        .function_step(
            function=lambda: record_func("start"),
        )
        .loop(condition="x < 3", args={"x": StepOutput("inside_loop")})
        .function_step(
            function=lambda: increment_counter("inside_loop"),
            step_name="inside_loop",
        )
        .end_loop()
        .function_step(
            function=lambda: record_func("end"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Should execute: start, inside_loop (3 times), end
    # String conditions are evaluated by the conditional evaluation agent
    # The exact behavior depends on the agent, but it should run the loop
    assert "start" in messages
    assert "end" in messages
    assert counter > 0


def test_plan_v2_loop_complex_nested_structure(local_portia: Portia) -> None:
    """Test PlanV2 complex nested structure with loops and conditionals."""
    messages: list[str] = []

    def record_func(message: str) -> None:
        messages.append(message)

    def generate_numbers() -> list[int]:
        return [1, 2, 3, 4, 5]

    def is_even(num: int) -> bool:
        return num % 2 == 0

    def is_positive(num: int) -> bool:
        return num > 0

    plan = (
        PlanBuilderV2(label="Test complex nested loops and conditionals")
        .function_step(
            function=lambda: record_func("start"),
        )
        .function_step(
            function=generate_numbers,
            step_name="numbers",
        )
        .loop(over=StepOutput("numbers"), step_name="Loop")
        .if_(condition=lambda num: is_even(num), args={"num": StepOutput("Loop")})
        .function_step(
            function=lambda num: record_func(f"even_number_{num}"),
            args={"num": StepOutput("Loop")},
        )
        .if_(condition=lambda num: is_positive(num), args={"num": StepOutput("Loop")})
        .function_step(
            function=lambda num: record_func(f"positive_even_{num}"),
            args={"num": StepOutput("Loop")},
        )
        .endif()
        .else_()
        .function_step(
            function=lambda num: record_func(f"odd_number_{num}"),
            args={"num": StepOutput("Loop")},
        )
        .endif()
        .end_loop()
        .function_step(
            function=lambda: record_func("end"),
        )
        .build()
    )

    plan_run = local_portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE

    # Should execute: start, then for each number 1-5:
    # - 1: odd_number_1
    # - 2: even_number_2, positive_even_2
    # - 3: odd_number_3
    # - 4: even_number_4, positive_even_4
    # - 5: odd_number_5
    assert "start" in messages
    assert "odd_number_1" in messages
    assert "even_number_2" in messages
    assert "positive_even_2" in messages
    assert "odd_number_3" in messages
    assert "even_number_4" in messages
    assert "positive_even_4" in messages
    assert "odd_number_5" in messages
    assert "end" in messages
