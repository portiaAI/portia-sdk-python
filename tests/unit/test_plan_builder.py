"""Test the PlanBuilder class."""

import pytest

from portia.plan import Plan, PlanBuilder, PlanContext, Step, Variable


def test_plan_builder_syntax() -> None:
    """Test that the plan builder syntax works."""
    plan_builder = (
        PlanBuilder("Find the best offers for a flight from London to New York")
        .step(
            task="Search for flights",
            tool_id="flight_search",
        )
        .step(
            task="compare prices",
            tool_id="price_comparison",
            condition="$flight_search has more than 30 results",
        )
        .input(
            name="$output_0",
        )
        .build()
    )
    base_plan = Plan(
        plan_context=PlanContext(
            query="Find the best offers for a flight from London to New York",
            tool_ids=["flight_search", "price_comparison"],
        ),
        steps=[
            Step(
                task="Search for flights",
                output="$output_0",
                tool_id="flight_search",
            ),
            Step(
                task="compare prices",
                output="$output_1",
                inputs=[Variable(name="$output_0", description="")],
                tool_id="price_comparison",
                condition="$flight_search has more than 30 results",
            ),
        ],
    )
    assert plan_builder.model_dump(exclude={"id"}) == base_plan.model_dump(exclude={"id"})


def test_plan_variable_no_steps() -> None:
    """Test that the plan variable function raises an error if there are no steps."""
    plan = PlanBuilder("Find the best offers for a flight from London to New York")
    with pytest.raises(ValueError, match="Invalid step index or no steps in the plan"):
        plan.input("$flights")


def test_plan_condition() -> None:
    """Test that the plan condition function works."""
    plan = (
        PlanBuilder("Find the best offers for a flight from London to New York")
        .step("search for flights", "flight_search")
        .step("compare prices", "price_comparison")
        .condition("The flight search has more than 30 results")
        .build()
    )
    assert plan.steps[1].condition == "The flight search has more than 30 results"
