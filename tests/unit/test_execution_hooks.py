"""Tests for execution hooks."""

from unittest.mock import MagicMock, patch

from portia.end_user import EndUser
from portia.execution_agents.output import LocalDataValue
from portia.plan import Plan, PlanContext, Step
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState
from portia.portia import ExecutionHooks, Portia
from portia.tool import ToolRunContext
from tests.utils import (
    AdditionTool,
    get_execution_ctx,
    get_test_config,
    get_test_plan_run,
)


class TestExecutionHooks:
    """Test class for ExecutionHooks."""

    def test_execution_step_hooks(self) -> None:
        """Test that execution step hooks are called at the appropriate times."""
        before_first_execution_step = MagicMock()
        before_execution_step = MagicMock()
        after_execution_step = MagicMock()
        after_last_execution_step = MagicMock()

        hooks = ExecutionHooks(
            before_first_execution_step=before_first_execution_step,
            before_execution_step=before_execution_step,
            after_execution_step=after_execution_step,
            after_last_execution_step=after_last_execution_step,
        )

        # Create a Portia instance with hooks
        config = get_test_config()
        portia = Portia(
            config=config,
            tools=[AdditionTool()],
            execution_hooks=hooks
        )

        step1 = Step(task="Step 1", output="$output1")
        step2 = Step(task="Step 2", output="$output2")
        plan = Plan(
            plan_context=PlanContext(
                query="Test query",
                tool_ids=["add_tool"],
            ),
            steps=[step1, step2],
        )

        plan_run = PlanRun(
            plan_id=plan.id,
            state=PlanRunState.NOT_STARTED,
            outputs=PlanRunOutputs(),
            end_user_id="test",
        )

        mock_agent = MagicMock()
        mock_agent.execute_sync.return_value = LocalDataValue(value="Test output")
        with patch.object(portia, "_get_agent_for_step", return_value=mock_agent):
            execute_plan_run = portia.__getattribute__("_execute_plan_run")
            execute_plan_run(plan, plan_run)

            before_first_execution_step.assert_called_once_with(plan, plan_run)

            assert before_execution_step.call_count == 2
            before_execution_step.assert_any_call(plan, plan_run, step1, 0)
            before_execution_step.assert_any_call(plan, plan_run, step2, 1)

            assert after_execution_step.call_count == 2
            args, _ = after_execution_step.call_args_list[0]
            assert args[0] == plan
            assert args[1] == plan_run
            assert args[2] == step1
            assert args[3] == 0
            assert isinstance(args[4], LocalDataValue)
            args, _ = after_execution_step.call_args_list[1]
            assert args[0] == plan
            assert args[1] == plan_run
            assert args[2] == step2
            assert args[3] == 1
            assert isinstance(args[4], LocalDataValue)

            after_last_execution_step.assert_called_once()
            args, _ = after_last_execution_step.call_args
            assert args[0] == plan
            assert args[1] == plan_run
            assert isinstance(args[2], LocalDataValue)

    def test_tool_call_hooks(self) -> None:
        """Test that tool call hooks are called at the appropriate times."""
        before_tool_call = MagicMock()
        after_tool_call = MagicMock()

        hooks = ExecutionHooks(
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
        )

        # Create a real config
        config = get_test_config()

        # Create a Portia instance with our hooks (used for context)

        tool = AdditionTool()
        plan, plan_run = get_test_plan_run()  # Unpack the tuple
        end_user = EndUser(external_id="test")

        # Create a real ToolRunContext
        tool_run_ctx = ToolRunContext(
            execution_context=get_execution_ctx(),
            end_user=end_user,
            plan_run_id=plan_run.id,
            config=config,
            clarifications=[],
        )

        from portia.execution_agents.default_execution_agent import DefaultExecutionAgent

        step = Step(task="Test step", output="$output", tool_id="add_tool")

        agent = DefaultExecutionAgent(
            step=step,
            plan_run=plan_run,
            config=config,
            agent_memory=MagicMock(),
            end_user=end_user,
            tool=tool,
            execution_hooks=hooks,
        )

        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "messages": [MagicMock(content='{"output": "Test result"}')]
        }

        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_app

        # Mock process_output to return a valid output
        mock_output = LocalDataValue(value="Test result")

        with (
            patch(
                "portia.execution_agents.default_execution_agent.StateGraph",
                return_value=mock_graph
            ),
            patch(
                "portia.execution_agents.default_execution_agent.get_execution_context",
                return_value={}
            ),
            patch(
                "portia.execution_agents.default_execution_agent.ToolRunContext",
                return_value=tool_run_ctx
            ),
            patch(
                "portia.execution_agents.default_execution_agent.process_output",
                return_value=mock_output
            ),
        ):
            result = agent.execute_sync()

            before_tool_call.assert_called_once()
            args, _ = before_tool_call.call_args
            assert args[0] == tool_run_ctx
            assert args[1] == tool

            after_tool_call.assert_called_once()
            args, _ = after_tool_call.call_args
            assert args[0] == tool_run_ctx
            assert args[1] == tool
            assert args[2] == result

        before_tool_call.reset_mock()
        after_tool_call.reset_mock()

        from portia.execution_agents.one_shot_agent import OneShotAgent

        agent = OneShotAgent(
            step=step,
            plan_run=plan_run,
            config=config,
            agent_memory=MagicMock(),
            end_user=end_user,
            tool=tool,
            execution_hooks=hooks,
        )

        with (
            patch(
                "portia.execution_agents.one_shot_agent.StateGraph",
                return_value=mock_graph
            ),
            patch(
                "portia.execution_agents.one_shot_agent.get_execution_context",
                return_value={}
            ),
            patch(
                "portia.execution_agents.one_shot_agent.ToolRunContext",
                return_value=tool_run_ctx
            ),
            patch(
                "portia.execution_agents.one_shot_agent.process_output",
                return_value=mock_output
            ),
        ):
            result = agent.execute_sync()

            before_tool_call.assert_called_once()
            args, _ = before_tool_call.call_args
            assert args[0] == tool_run_ctx
            assert args[1] == tool

            after_tool_call.assert_called_once()
            args, _ = after_tool_call.call_args
            assert args[0] == tool_run_ctx
            assert args[1] == tool
            assert args[2] == result
