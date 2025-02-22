import pytest

from portia.plan import Plan, PlanBuilder, PlanContext, Step, Variable


def test_plan_builder_syntax() -> None:
    """Test that the plan builder syntax works."""
    plan_builder = (
        PlanBuilder("Find the best offers for a flight from London to New York")
        .step(
            "Search for flights",
            "flight_search",
        )
        .step(
            "compare prices",
            "price_comparison",
        )
        .input(
            "$output_0",
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
                inputs=[Variable(name="$output_0", value=None, description="")],
                tool_id="price_comparison",
            ),
        ],
    )
    assert plan_builder.model_dump(exclude={"id"}) == base_plan.model_dump(exclude={"id"})


def test_plan_variable_no_steps() -> None:
    """Test that the plan variable function raises an error if there are no steps."""
    plan = PlanBuilder("Find the best offers for a flight from London to New York")
    with pytest.raises(ValueError, match="No steps in the plan"):
        plan.input("$flights")
