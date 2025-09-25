"""Tests for execution state dataclasses."""

from __future__ import annotations

from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue
from portia.execution_state import (
    IntrospectionResult,
    PlanRunReadinessResult,
    PlanRunSession,
    PlanTestBundle,
)
from portia.introspection_agents.introspection_agent import (
    PreStepIntrospection,
    PreStepIntrospectionOutcome,
)
from portia.plan import Plan, PlanContext, Step, Variable
from portia.plan_run import PlanRun, PlanRunState


class TestPlanRunReadinessResult:
    """Test PlanRunReadinessResult dataclass."""

    def test_initialization(self) -> None:
        """Test basic initialization of PlanRunReadinessResult."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        result = PlanRunReadinessResult(is_ready=True, plan_run=plan_run)

        assert result.is_ready is True
        assert result.plan_run is plan_run

    def test_not_ready_state(self) -> None:
        """Test PlanRunReadinessResult with not ready state."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        plan_run.state = PlanRunState.FAILED
        result = PlanRunReadinessResult(is_ready=False, plan_run=plan_run)

        assert result.is_ready is False
        assert result.plan_run.state == PlanRunState.FAILED

    def test_serialization(self) -> None:
        """Test that PlanRunReadinessResult can be serialized."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        result = PlanRunReadinessResult(is_ready=True, plan_run=plan_run)

        # Test that it can be converted to dict
        data = result.model_dump()
        assert data["is_ready"] is True
        assert data["plan_run"] is not None

        # Test that it can be recreated from dict
        recreated = PlanRunReadinessResult.model_validate(data)
        assert recreated.is_ready == result.is_ready
        assert recreated.plan_run.plan_id == result.plan_run.plan_id


class TestIntrospectionResult:
    """Test IntrospectionResult dataclass."""

    def test_initialization(self) -> None:
        """Test basic initialization of IntrospectionResult."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        introspection = PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.CONTINUE,
            reason="All conditions met"
        )
        result = IntrospectionResult(plan_run=plan_run, introspection=introspection)

        assert result.plan_run is plan_run
        assert result.introspection is introspection
        assert result.introspection.outcome == PreStepIntrospectionOutcome.CONTINUE
        assert result.introspection.reason == "All conditions met"

    def test_skip_outcome(self) -> None:
        """Test IntrospectionResult with SKIP outcome."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        introspection = PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.SKIP,
            reason="Condition not met"
        )
        result = IntrospectionResult(plan_run=plan_run, introspection=introspection)

        assert result.introspection.outcome == PreStepIntrospectionOutcome.SKIP
        assert result.introspection.reason == "Condition not met"

    def test_complete_outcome(self) -> None:
        """Test IntrospectionResult with COMPLETE outcome."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        introspection = PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.COMPLETE,
            reason="Plan completed early"
        )
        result = IntrospectionResult(plan_run=plan_run, introspection=introspection)

        assert result.introspection.outcome == PreStepIntrospectionOutcome.COMPLETE
        assert result.introspection.reason == "Plan completed early"

    def test_serialization(self) -> None:
        """Test that IntrospectionResult can be serialized."""
        plan_run = PlanRun(plan_id="test_plan", end_user_id="test_user")
        introspection = PreStepIntrospection(
            outcome=PreStepIntrospectionOutcome.CONTINUE,
            reason="Test reason"
        )
        result = IntrospectionResult(plan_run=plan_run, introspection=introspection)

        # Test that it can be converted to dict
        data = result.model_dump()
        assert data["plan_run"] is not None
        assert data["introspection"] is not None

        # Test that it can be recreated from dict
        recreated = IntrospectionResult.model_validate(data)
        assert recreated.introspection.outcome == result.introspection.outcome
        assert recreated.introspection.reason == result.introspection.reason
        assert recreated.plan_run.plan_id == result.plan_run.plan_id


class TestPlanRunSession:
    """Test PlanRunSession dataclass."""

    def test_initialization_minimal(self) -> None:
        """Test basic initialization of PlanRunSession."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")
        end_user = EndUser(external_id="test_user", name="Test User")

        session = PlanRunSession(plan=plan, plan_run=plan_run, end_user=end_user)

        assert session.plan is plan
        assert session.plan_run is plan_run
        assert session.end_user is end_user
        assert session.last_step_output is None

    def test_initialization_with_output(self) -> None:
        """Test initialization of PlanRunSession with step output."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")
        end_user = EndUser(external_id="test_user", name="Test User")
        step_output = LocalDataValue(value="test result")

        session = PlanRunSession(
            plan=plan,
            plan_run=plan_run,
            end_user=end_user,
            last_step_output=step_output
        )

        assert session.last_step_output is step_output
        assert session.last_step_output.value == "test result"

    def test_create_classmethod(self) -> None:
        """Test the create classmethod."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")
        end_user = EndUser(external_id="test_user", name="Test User")

        session = PlanRunSession.create(plan=plan, plan_run=plan_run, end_user=end_user)

        assert session.plan is plan
        assert session.plan_run is plan_run
        assert session.end_user is end_user
        assert session.last_step_output is None

    def test_create_with_output(self) -> None:
        """Test the create classmethod with step output."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")
        end_user = EndUser(external_id="test_user", name="Test User")
        step_output = LocalDataValue(value="test result")

        session = PlanRunSession.create(
            plan=plan,
            plan_run=plan_run,
            end_user=end_user,
            last_step_output=step_output
        )

        assert session.last_step_output is step_output

    def test_serialization(self) -> None:
        """Test that PlanRunSession can be serialized."""
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")
        end_user = EndUser(external_id="test_user", name="Test User")

        session = PlanRunSession(plan=plan, plan_run=plan_run, end_user=end_user)

        # Test that it can be converted to dict
        data = session.model_dump()
        assert data["plan"] is not None
        assert data["plan_run"] is not None
        assert data["end_user"] is not None
        assert data["last_step_output"] is None

        # Test that it can be recreated from dict
        recreated = PlanRunSession.model_validate(data)
        assert recreated.plan.plan_context.query == session.plan.plan_context.query
        assert recreated.plan_run.plan_id == session.plan_run.plan_id
        assert recreated.end_user.external_id == session.end_user.external_id


class TestPlanTestBundle:
    """Test PlanTestBundle dataclass."""

    def test_initialization(self) -> None:
        """Test basic initialization of PlanTestBundle."""
        step = Step(
            task="Add numbers",
            inputs=[Variable(name="$a", description="first number")],
            output="$result"
        )
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[step]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")

        bundle = PlanTestBundle(plan=plan, plan_run=plan_run)

        assert bundle.plan is plan
        assert bundle.plan_run is plan_run
        assert len(bundle.plan.steps) == 1
        assert bundle.plan.steps[0].task == "Add numbers"

    def test_unpack_method(self) -> None:
        """Test the unpack method for backwards compatibility."""
        step = Step(
            task="Add numbers",
            inputs=[Variable(name="$a", description="first number")],
            output="$result"
        )
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[step]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")

        bundle = PlanTestBundle(plan=plan, plan_run=plan_run)
        unpacked_plan, unpacked_plan_run = bundle.unpack()

        assert unpacked_plan is plan
        assert unpacked_plan_run is plan_run

    def test_serialization(self) -> None:
        """Test that PlanTestBundle can be serialized."""
        step = Step(
            task="Add numbers",
            inputs=[Variable(name="$a", description="first number")],
            output="$result"
        )
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[step]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")

        bundle = PlanTestBundle(plan=plan, plan_run=plan_run)

        # Test that it can be converted to dict
        data = bundle.model_dump()
        assert data["plan"] is not None
        assert data["plan_run"] is not None

        # Test that it can be recreated from dict
        recreated = PlanTestBundle.model_validate(data)
        assert recreated.plan.plan_context.query == bundle.plan.plan_context.query
        assert recreated.plan_run.plan_id == bundle.plan_run.plan_id
        assert len(recreated.plan.steps) == 1
        assert recreated.plan.steps[0].task == "Add numbers"

    def test_with_step_outputs(self) -> None:
        """Test PlanTestBundle with step outputs."""
        step = Step(
            task="Add numbers",
            inputs=[Variable(name="$a", description="first number")],
            output="$result"
        )
        plan = Plan(
            plan_context=PlanContext(query="Test query", tool_ids=["test_tool"]),
            steps=[step]
        )
        plan_run = PlanRun(plan_id=plan.id, end_user_id="test_user")
        plan_run.outputs.step_outputs = {
            "$a": LocalDataValue(value="5"),
            "$result": LocalDataValue(value="7")
        }

        bundle = PlanTestBundle(plan=plan, plan_run=plan_run)

        assert "$a" in bundle.plan_run.outputs.step_outputs
        assert "$result" in bundle.plan_run.outputs.step_outputs
        assert bundle.plan_run.outputs.step_outputs["$a"].value == "5"
        assert bundle.plan_run.outputs.step_outputs["$result"].value == "7"
