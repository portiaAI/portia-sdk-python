"""Integration tests for RunContext with PortiaContext."""

from unittest.mock import Mock

import pytest

from portia.builder.plan_v2 import PlanV2
from portia.config import Config
from portia.core.context import PortiaContext
from portia.end_user import EndUser
from portia.execution_hooks import ExecutionHooks
from portia.logger import logger_manager
from portia.plan import Plan
from portia.plan_run import PlanRun
from portia.run_context import RunContext, StepOutputValue
from portia.storage import InMemoryStorage
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.tool_registry import ToolRegistry


class TestRunContextIntegration:
    """Integration tests for RunContext with PortiaContext."""

    @pytest.fixture
    def portia_context(self):
        """Create a PortiaContext for testing."""
        return PortiaContext(
            config=Config.from_default(),
            logger_manager=logger_manager,
            storage=InMemoryStorage(),
            tool_registry=ToolRegistry([]),
            telemetry=ProductTelemetry(),
            execution_hooks=ExecutionHooks(),
        )

    @pytest.fixture
    def run_context_deps(self):
        """Create dependencies for RunContext."""
        return {
            "plan": Mock(spec=PlanV2),
            "legacy_plan": Mock(spec=Plan),
            "plan_run": Mock(spec=PlanRun),
            "end_user": Mock(spec=EndUser),
            "step_output_values": [],
        }

    def test_run_context_with_portia_context(self, portia_context, run_context_deps):
        """Test that RunContext works with PortiaContext."""
        run_context_deps["context"] = portia_context
        run_context = RunContext(**run_context_deps)

        assert run_context.context is portia_context
        assert run_context.plan is run_context_deps["plan"]
        assert run_context.legacy_plan is run_context_deps["legacy_plan"]
        assert run_context.plan_run is run_context_deps["plan_run"]
        assert run_context.end_user is run_context_deps["end_user"]

    def test_run_context_tool_run_context_integration(self, portia_context, run_context_deps):
        """Test that RunContext.get_tool_run_ctx() works with PortiaContext."""
        # Mock the plan_run to return empty clarifications
        run_context_deps["plan_run"].get_clarifications_for_step.return_value = []
        run_context_deps["context"] = portia_context

        run_context = RunContext(**run_context_deps)
        tool_run_ctx = run_context.get_tool_run_ctx()

        # Verify that the tool run context was created with config from the context
        assert tool_run_ctx.config is portia_context.config
        assert tool_run_ctx.end_user is run_context_deps["end_user"]
        assert tool_run_ctx.plan_run is run_context_deps["plan_run"]
        assert tool_run_ctx.plan is run_context_deps["legacy_plan"]

    def test_run_context_step_output_values(self, portia_context, run_context_deps):
        """Test that step output values work correctly in RunContext."""
        step_outputs = [
            StepOutputValue(
                value="output1", description="First output", step_name="step1", step_num=0
            ),
            StepOutputValue(
                value="output2", description="Second output", step_name="step2", step_num=1
            ),
        ]

        run_context_deps["context"] = portia_context
        run_context_deps["step_output_values"] = step_outputs

        run_context = RunContext(**run_context_deps)

        assert len(run_context.step_output_values) == 2
        assert run_context.step_output_values[0].value == "output1"
        assert run_context.step_output_values[1].value == "output2"

    def test_run_context_backward_compatibility(self, portia_context, run_context_deps):
        """Test that RunContext maintains backward compatibility for field access."""
        run_context_deps["context"] = portia_context
        run_context = RunContext(**run_context_deps)

        # These should work through the context
        assert run_context.context.config is not None
        assert run_context.context.storage is not None
        assert run_context.context.tool_registry is not None
        assert run_context.context.telemetry is not None
        assert run_context.context.execution_hooks is not None

    def test_run_context_pydantic_validation(self, portia_context, run_context_deps):
        """Test that RunContext properly validates fields through Pydantic."""
        run_context_deps["context"] = portia_context

        # Should create successfully with all required fields
        run_context = RunContext(**run_context_deps)
        assert isinstance(run_context, RunContext)

        # Should fail validation without required fields
        with pytest.raises(ValueError, match="Field required"):
            RunContext(context=portia_context)  # Missing other required fields

    def test_run_context_model_config(self, portia_context, run_context_deps):
        """Test that RunContext model config allows arbitrary types."""
        run_context_deps["context"] = portia_context
        run_context = RunContext(**run_context_deps)

        # Verify that the model config allows arbitrary types
        assert run_context.model_config.arbitrary_types_allowed

        # This should work because arbitrary types are allowed
        custom_object = Mock()
        run_context_deps["plan"] = custom_object
        run_context_with_custom = RunContext(**run_context_deps)
        assert run_context_with_custom.plan is custom_object