"""Integration tests for PlanV2 examples."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from portia import Config, LogLevel, Portia
from portia.builder.plan_builder import PlanBuilderError, PlanBuilderV2
from portia.builder.reference import Input, StepOutput
from portia.config import StorageClass
from portia.plan_run import PlanRunState
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
