"""Tests for runner classes."""

import tempfile
import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl

from portia.agents.base_agent import Output
from portia.clarification import ActionClarification, InputClarification
from portia.config import Config, StorageClass
from portia.errors import InvalidWorkflowStateError, PlanError, WorkflowNotFoundError
from portia.llm_wrapper import LLMWrapper
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.registry import example_tool_registry, open_source_tool_registry
from portia.plan import Plan, PlanContext, ReadOnlyPlan, Step
from portia.planners.planner import StepsOrError
from portia.runner import Runner
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import ReadOnlyWorkflow, Workflow, WorkflowOutputs, WorkflowState, WorkflowUUID
from tests.utils import AdditionTool, ClarificationTool, get_test_config, get_test_workflow


@pytest.fixture
def runner() -> Runner:
    """Fixture to create a Runner instance for testing."""
    config = get_test_config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
    return Runner(config=config, tools=tool_registry)


def test_runner_local_default_config_with_api_keys() -> None:
    """Test that the default config is used if no config is provided."""
    # Unset the portia API env that the runner doesn't try to use Portia Cloud
    with mock.patch.dict(
        "os.environ",
        {
            "PORTIA_API_KEY": "",
            "OPENAI_API_KEY": "123",
            "TAVILY_API_KEY": "123",
            "OPENWEATHERMAP_API_KEY": "123",
        },
    ):
        runner = Runner()
        assert runner.config == Config.from_default()
        assert len(runner.tool_registry.get_tools()) == len(open_source_tool_registry.get_tools())


def test_runner_local_default_config_without_api_keys() -> None:
    """Test that the default config when no API keys are provided."""
    # Unset the Tavily and weather API and check that these aren't included in
    # the default tool registry
    with mock.patch.dict(
        "os.environ",
        {
            "PORTIA_API_KEY": "",
            "OPENAI_API_KEY": "123",
            "TAVILY_API_KEY": "",
            "OPENWEATHERMAP_API_KEY": "",
        },
    ):
        runner = Runner()
        assert runner.config == Config.from_default()
        assert (
            len(runner.tool_registry.get_tools()) == len(open_source_tool_registry.get_tools()) - 2
        )


def test_runner_run_query(runner: Runner) -> None:
    """Test running a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    workflow = runner.execute_query(query)

    assert workflow.state == WorkflowState.COMPLETE


def test_runner_run_query_tool_list() -> None:
    """Test running a query using the Runner."""
    query = "example query"
    runner = Runner(config=get_test_config(), tools=[AdditionTool(), ClarificationTool()])

    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    workflow = runner.execute_query(query)

    assert workflow.state == WorkflowState.COMPLETE


def test_runner_run_query_disk_storage() -> None:
    """Test running a query using the Runner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        query = "example query"
        config = get_test_config(
            storage_class=StorageClass.DISK,
            storage_dir=tmp_dir,
        )
        tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
        runner = Runner(config=config, tools=tool_registry)

        mock_response = StepsOrError(steps=[], error=None)
        LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

        workflow = runner.execute_query(query)

        assert workflow.state == WorkflowState.COMPLETE
        # Use Path to check for the files
        plan_files = list(Path(tmp_dir).glob("plan-*.json"))
        workflow_files = list(Path(tmp_dir).glob("wkfl-*.json"))

        assert len(plan_files) == 1
        assert len(workflow_files) == 1


def test_runner_generate_plan(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)

    assert plan.plan_context.query == query


def test_runner_generate_plan_error(runner: Runner) -> None:
    """Test planning a query that returns an error."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error="could not plan")
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    with pytest.raises(PlanError):
        runner.generate_plan(query)


def test_runner_generate_plan_with_tools(runner: Runner) -> None:
    """Test planning a query using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query, tools=["add_tool"])

    assert plan.plan_context.query == query
    assert plan.plan_context.tool_ids == ["add_tool"]


def test_runner_create_and_execute_workflow(runner: Runner) -> None:
    """Test running a plan using the Runner."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id == plan.id


def test_runner_execute_workflow(runner: Runner) -> None:
    """Test resuming a workflow after interruption."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    # Simulate workflow being in progress
    workflow.state = WorkflowState.IN_PROGRESS
    workflow.current_step_index = 1
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.current_step_index == 1


def test_runner_execute_workflow_edge_cases(runner: Runner) -> None:
    """Test edge cases for execute."""
    with pytest.raises(ValueError):  # noqa: PT011
        runner.execute_workflow()

    query = "example query"
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)

    # Simulate workflow being in progress
    workflow.state = WorkflowState.IN_PROGRESS
    workflow.current_step_index = 1
    workflow = runner.execute_workflow(workflow_id=workflow.id)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.current_step_index == 1

    with pytest.raises(WorkflowNotFoundError):
        runner.execute_workflow(workflow_id=WorkflowUUID())


def test_runner_execute_workflow_invalid_state(runner: Runner) -> None:
    """Test resuming a workflow with an invalid state."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    # Set invalid state
    workflow.state = WorkflowState.COMPLETE

    with pytest.raises(InvalidWorkflowStateError):
        runner.execute_workflow(workflow)


def test_runner_wait_for_ready(runner: Runner) -> None:
    """Test wait for ready."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = runner.generate_plan(query)
    workflow = runner.create_workflow(plan)

    workflow.state = WorkflowState.FAILED
    with pytest.raises(InvalidWorkflowStateError):
        runner.wait_for_ready(workflow)

    workflow.state = WorkflowState.IN_PROGRESS
    workflow = runner.wait_for_ready(workflow)
    assert workflow.state == WorkflowState.IN_PROGRESS

    def update_workflow_state() -> None:
        """Update the workflow state after sleeping."""
        time.sleep(1)  # Simulate some delay before state changes
        workflow.state = WorkflowState.READY_TO_RESUME
        runner.storage.save_workflow(workflow)

    workflow.state = WorkflowState.NEED_CLARIFICATION

    # start a thread to update in status
    update_thread = threading.Thread(target=update_workflow_state)
    update_thread.start()

    workflow = runner.wait_for_ready(workflow)
    assert workflow.state == WorkflowState.READY_TO_RESUME


def test_runner_wait_for_ready_tool(runner: Runner) -> None:
    """Test wait for ready."""
    mock_call_count = MagicMock()
    mock_call_count.__iadd__ = (
        lambda self, other: setattr(self, "count", self.count + other) or self
    )
    mock_call_count.count = 0

    class ReadyTool(Tool):
        """Returns ready."""

        id: str = "ready_tool"
        name: str = "Ready Tool"
        description: str = "Returns a clarification"
        output_schema: tuple[str, str] = (
            "Clarification",
            "Clarification: The value of the Clarification",
        )

        def run(self, ctx: ToolRunContext, user_guidance: str) -> str:  # noqa: ARG002
            return "result"

        def ready(self, ctx: ToolRunContext) -> bool:  # noqa: ARG002
            mock_call_count.count += 1
            return mock_call_count.count == 3

    runner.tool_registry = InMemoryToolRegistry.from_local_tools([ReadyTool()])
    step0 = Step(
        task="Do something",
        inputs=[],
        output="$ctx_0",
    )
    step1 = Step(
        task="Save Context",
        inputs=[],
        output="$ctx",
        tool_id="ready_tool",
    )
    plan = Plan(
        plan_context=PlanContext(
            query="run the tool",
            tool_ids=["ready_tool"],
        ),
        steps=[step0, step1],
    )
    unresolved_action = ActionClarification(
        workflow_id=WorkflowUUID(),
        user_guidance="",
        action_url=HttpUrl("https://unresolved.step1.com"),
        step=1,
    )
    workflow = Workflow(
        plan_id=plan.id,
        current_step_index=1,
        state=WorkflowState.NEED_CLARIFICATION,
        outputs=WorkflowOutputs(
            clarifications=[
                ActionClarification(
                    workflow_id=WorkflowUUID(),
                    user_guidance="",
                    action_url=HttpUrl("https://resolved.step0.com"),
                    resolved=True,
                    step=0,
                ),
                ActionClarification(
                    workflow_id=WorkflowUUID(),
                    user_guidance="",
                    action_url=HttpUrl("https://resolved.step1.com"),
                    resolved=True,
                    step=1,
                ),
                unresolved_action,
            ],
        ),
    )
    runner.storage.save_plan(plan)
    runner.storage.save_workflow(workflow)
    assert workflow.get_outstanding_clarifications() == [unresolved_action]
    workflow = runner.wait_for_ready(workflow)
    assert workflow.state == WorkflowState.READY_TO_RESUME
    assert workflow.get_outstanding_clarifications() == []
    for clarification in workflow.outputs.clarifications:
        if clarification.step == 1:
            assert clarification.resolved
            assert clarification.response == "complete"


def test_runner_execute_query_with_summary(runner: Runner) -> None:
    """Test execute_query sets both final output and summary correctly."""
    query = "What activities can I do in London based on weather?"

    # Mock planner response
    weather_step = Step(
        task="Get weather in London",
        tool_id="add_tool",
        output="$weather",
    )
    activities_step = Step(
        task="Suggest activities based on weather",
        tool_id="add_tool",
        output="$activities",
    )
    mock_plan = StepsOrError(
        steps=[weather_step, activities_step],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_plan)

    # Mock agent responses
    weather_output = Output(value="Sunny and warm")
    activities_output = Output(value="Visit Hyde Park and have a picnic")
    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"

    mock_step_agent = mock.MagicMock()
    mock_step_agent.execute_sync.side_effect = [weather_output, activities_output]

    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.side_effect = [expected_summary]

    with (
        mock.patch(
            "portia.runner.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(runner, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        workflow = runner.execute_query(query)

        # Verify workflow completed successfully
        assert workflow.state == WorkflowState.COMPLETE

        # Verify step outputs were stored correctly
        assert workflow.outputs.step_outputs["$weather"] == weather_output
        assert workflow.outputs.step_outputs["$activities"] == activities_output

        # Verify final output and summary
        assert workflow.outputs.final_output is not None
        assert workflow.outputs.final_output.value == activities_output.value
        assert workflow.outputs.final_output.summary == expected_summary

        # Verify create_summary was called with correct args
        mock_summarizer_agent.create_summary.assert_called_once_with(
            plan=mock.ANY,
            workflow=mock.ANY,
        )


def test_runner_sets_final_output_with_summary(runner: Runner) -> None:
    """Test that final output is set with correct summary."""
    (plan, workflow) = get_test_workflow()
    plan.steps = [
        Step(
            task="Get weather in London",
            output="$london_weather",
        ),
        Step(
            task="Suggest activities based on weather",
            output="$activities",
        ),
    ]

    workflow.outputs.step_outputs = {
        "$london_weather": Output(value="Sunny and warm"),
        "$activities": Output(value="Visit Hyde Park and have a picnic"),
    }

    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"
    mock_summarizer = mock.MagicMock()
    mock_summarizer.create_summary.side_effect = [expected_summary]

    with mock.patch(
        "portia.runner.FinalOutputSummarizer",
        return_value=mock_summarizer,
    ):
        last_step_output = Output(value="Visit Hyde Park and have a picnic")
        output = runner._get_final_output(plan, workflow, last_step_output)  # noqa: SLF001

        # Verify the final output
        assert output is not None
        assert output.value == "Visit Hyde Park and have a picnic"
        assert output.summary == expected_summary

        # Verify create_summary was called with correct args
        mock_summarizer.create_summary.assert_called_once()
        call_args = mock_summarizer.create_summary.call_args[1]
        assert isinstance(call_args["plan"], ReadOnlyPlan)
        assert isinstance(call_args["workflow"], ReadOnlyWorkflow)
        assert call_args["plan"].id == plan.id
        assert call_args["workflow"].id == workflow.id


def test_runner_get_final_output_handles_summary_error(runner: Runner) -> None:
    """Test that final output is set even if summary generation fails."""
    (plan, workflow) = get_test_workflow()

    # Mock the SummarizerAgent to raise an exception
    mock_agent = mock.MagicMock()
    mock_agent.create_summary.side_effect = Exception("Summary failed")

    with mock.patch(
        "portia.agents.utils.final_output_summarizer.FinalOutputSummarizer",
        return_value=mock_agent,
    ):
        step_output = Output(value="Some output")
        final_output = runner._get_final_output(plan, workflow, step_output)  # noqa: SLF001

        # Verify the final output is set without summary
        assert final_output is not None
        assert final_output.value == "Some output"
        assert final_output.summary is None


def test_runner_wait_for_ready_max_retries(runner: Runner) -> None:
    """Test wait for ready with max retries."""
    plan, workflow = get_test_workflow()
    workflow.state = WorkflowState.NEED_CLARIFICATION
    runner.storage.save_plan(plan)
    with pytest.raises(InvalidWorkflowStateError):
        runner.wait_for_ready(workflow, max_retries=0)


def test_runner_wait_for_ready_backoff_period(runner: Runner) -> None:
    """Test wait for ready with backoff period."""
    plan, workflow = get_test_workflow()
    workflow.state = WorkflowState.NEED_CLARIFICATION
    runner.storage.save_plan(plan)
    runner.storage.get_workflow = mock.MagicMock(return_value=workflow)
    with pytest.raises(InvalidWorkflowStateError):
        runner.wait_for_ready(workflow, max_retries=1, backoff_start_time_seconds=0)


def test_runner_resolve_clarification_error(runner: Runner) -> None:
    """Test resolve error."""
    plan, workflow = get_test_workflow()
    plan2, workflow2 = get_test_workflow()
    clarification = InputClarification(
        user_guidance="",
        argument_name="",
        workflow_id=workflow2.id,
    )
    runner.storage.save_plan(plan)
    runner.storage.save_workflow(workflow)
    runner.storage.save_plan(plan2)
    runner.storage.save_workflow(workflow2)
    with pytest.raises(InvalidWorkflowStateError):
        runner.resolve_clarification(clarification, "test")

    with pytest.raises(InvalidWorkflowStateError):
        runner.resolve_clarification(clarification, "test", workflow)


def test_runner_resolve_clarification(runner: Runner) -> None:
    """Test resolve success."""
    plan, workflow = get_test_workflow()
    clarification = InputClarification(
        user_guidance="",
        argument_name="",
        workflow_id=workflow.id,
    )
    workflow.outputs.clarifications = [clarification]
    runner.storage.save_plan(plan)
    runner.storage.save_workflow(workflow)

    workflow = runner.resolve_clarification(clarification, "test", workflow)
    assert workflow.state == WorkflowState.READY_TO_RESUME


def test_runner_get_tool_for_step_none_tool_id() -> None:
    """Test that when step.tool_id is None, LLMTool is used as fallback."""
    runner = Runner(config=get_test_config(), tools=[AdditionTool()])
    plan, workflow = get_test_workflow()

    # Create a step with no tool_id
    step = Step(
        task="Some task",
        inputs=[],
        output="$output",
        tool_id=None,
    )

    tool = runner._get_tool_for_step(step, workflow)  # noqa: SLF001
    assert tool is None


def test_get_llm_tool() -> None:
    """Test special case retrieval of LLMTool as it isn't explicitly in most tool registries."""
    runner = Runner(config=get_test_config(), tools=example_tool_registry)
    plan, workflow = get_test_workflow()

    # Create a step with no tool_id
    step = Step(
        task="Some task",
        inputs=[],
        output="$output",
        tool_id=LLMTool.LLM_TOOL_ID,
    )

    tool = runner._get_tool_for_step(step, workflow)  # noqa: SLF001
    assert tool is not None
    assert isinstance(tool._child_tool, LLMTool)  # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]
