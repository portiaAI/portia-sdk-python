"""Tests for portia classes."""

import tempfile
import threading
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl, SecretStr

from portia.clarification import (
    ActionClarification,
    InputClarification,
    ValueConfirmationClarification,
)
from portia.config import Config, StorageClass
from portia.errors import InvalidPlanRunStateError, PlanError, PlanRunNotFoundError
from portia.execution_agents.base_execution_agent import Output
from portia.llm_wrapper import LLMWrapper
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.registry import example_tool_registry, open_source_tool_registry
from portia.plan import Plan, PlanContext, ReadOnlyPlan, Step
from portia.plan_run import PlanRun, PlanRunOutputs, PlanRunState, PlanRunUUID, ReadOnlyPlanRun
from portia.planning_agents.base_planning_agent import StepsOrError
from portia.portia import ExecutionHooks, Portia
from portia.tool import Tool, ToolRunContext
from portia.tool_registry import InMemoryToolRegistry
from tests.utils import (
    AdditionTool,
    ClarificationTool,
    TestClarificationHandler,
    get_test_config,
    get_test_plan_run,
)


@pytest.fixture
def portia() -> Portia:
    """Fixture to create a Portia instance for testing."""
    config = get_test_config()
    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
    return Portia(config=config, tools=tool_registry)


def test_portia_local_default_config_with_api_keys() -> None:
    """Test that the default config is used if no config is provided."""
    # Unset the portia API env that the portia doesn't try to use Portia Cloud
    with mock.patch.dict(
        "os.environ",
        {
            "PORTIA_API_KEY": "",
            "OPENAI_API_KEY": "123",
            "TAVILY_API_KEY": "123",
            "OPENWEATHERMAP_API_KEY": "123",
        },
    ):
        portia = Portia()
        assert portia.config == Config.from_default()
        assert len(portia.tool_registry.get_tools()) == len(open_source_tool_registry.get_tools())


def test_portia_local_default_config_without_api_keys() -> None:
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
        portia = Portia()
        assert portia.config == Config.from_default()
        assert (
            len(portia.tool_registry.get_tools()) == len(open_source_tool_registry.get_tools()) - 2
        )


def test_portia_run_query(portia: Portia) -> None:
    """Test running a query."""
    query = "example query"

    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_query_tool_list() -> None:
    """Test running a query."""
    query = "example query"
    portia = Portia(config=get_test_config(), tools=[AdditionTool(), ClarificationTool()])

    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_query_disk_storage() -> None:
    """Test running a query."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        query = "example query"
        config = Config.from_default(
            storage_class=StorageClass.DISK,
            openai_api_key=SecretStr("123"),
            storage_dir=tmp_dir,
        )
        tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool(), ClarificationTool()])
        portia = Portia(config=config, tools=tool_registry)

        mock_response = StepsOrError(steps=[], error=None)
        LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

        plan_run = portia.run(query)

        assert plan_run.state == PlanRunState.COMPLETE
        # Use Path to check for the files
        plan_files = list(Path(tmp_dir).glob("plan-*.json"))
        run_files = list(Path(tmp_dir).glob("prun-*.json"))

        assert len(plan_files) == 1
        assert len(run_files) == 1


def test_portia_generate_plan(portia: Portia) -> None:
    """Test planning a query."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = portia.plan(query)

    assert plan.plan_context.query == query


def test_portia_generate_plan_error(portia: Portia) -> None:
    """Test planning a query that returns an error."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error="could not plan")
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    with pytest.raises(PlanError):
        portia.plan(query)


def test_portia_generate_plan_with_tools(portia: Portia) -> None:
    """Test planning a query."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = portia.plan(query, tools=["add_tool"])

    assert plan.plan_context.query == query
    assert plan.plan_context.tool_ids == ["add_tool"]


def test_portia_resume(portia: Portia) -> None:
    """Test running a plan."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = portia.plan(query)
    plan_run = portia._create_plan_run(plan)  # noqa: SLF001
    plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.plan_id == plan.id


def test_portia_resume_after_interruption(portia: Portia) -> None:
    """Test resuming PlanRun after interruption."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run(query)

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1
    plan_run = portia.resume(plan_run)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1


def test_portia_resume_edge_cases(portia: Portia) -> None:
    """Test edge cases for execute."""
    with pytest.raises(ValueError):  # noqa: PT011
        portia.resume()

    query = "example query"
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = portia.plan(query)
    plan_run = portia._create_plan_run(plan)  # noqa: SLF001

    # Simulate run being in progress
    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run.current_step_index = 1
    plan_run = portia.resume(plan_run_id=plan_run.id)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.current_step_index == 1

    with pytest.raises(PlanRunNotFoundError):
        portia.resume(plan_run_id=PlanRunUUID())


def test_portia_run_invalid_state(portia: Portia) -> None:
    """Test resuming PlanRun with an invalid state."""
    query = "example query"

    mock_response = StepsOrError(steps=[], error=None)
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run(query)

    # Set invalid state
    plan_run.state = PlanRunState.COMPLETE

    with pytest.raises(InvalidPlanRunStateError):
        portia.resume(plan_run)


def test_portia_wait_for_ready(portia: Portia) -> None:
    """Test wait for ready."""
    query = "example query"

    mock_response = StepsOrError(
        steps=[Step(task="Example task", inputs=[], output="$output")],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run(query)

    plan_run.state = PlanRunState.FAILED
    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run)

    plan_run.state = PlanRunState.IN_PROGRESS
    plan_run = portia.wait_for_ready(plan_run)
    assert plan_run.state == PlanRunState.IN_PROGRESS

    def update_run_state() -> None:
        """Update the run state after sleeping."""
        time.sleep(1)  # Simulate some delay before state changes
        plan_run.state = PlanRunState.READY_TO_RESUME
        portia.storage.save_plan_run(plan_run)

    plan_run.state = PlanRunState.NEED_CLARIFICATION

    # Ensure current_step_index is set to a valid index
    plan_run.current_step_index = 0
    portia.storage.save_plan_run(plan_run)

    # start a thread to update in status
    update_thread = threading.Thread(target=update_run_state)
    update_thread.start()

    plan_run = portia.wait_for_ready(plan_run)
    assert plan_run.state == PlanRunState.READY_TO_RESUME


def test_portia_wait_for_ready_tool(portia: Portia) -> None:
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

    portia.tool_registry = InMemoryToolRegistry.from_local_tools([ReadyTool()])
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
        plan_run_id=PlanRunUUID(),
        user_guidance="",
        action_url=HttpUrl("https://unresolved.step1.com"),
        step=1,
    )
    plan_run = PlanRun(
        plan_id=plan.id,
        current_step_index=1,
        state=PlanRunState.NEED_CLARIFICATION,
        outputs=PlanRunOutputs(
            clarifications=[
                ActionClarification(
                    plan_run_id=PlanRunUUID(),
                    user_guidance="",
                    action_url=HttpUrl("https://resolved.step0.com"),
                    resolved=True,
                    step=0,
                ),
                ActionClarification(
                    plan_run_id=PlanRunUUID(),
                    user_guidance="",
                    action_url=HttpUrl("https://resolved.step1.com"),
                    resolved=True,
                    step=1,
                ),
                unresolved_action,
            ],
        ),
    )
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)
    assert plan_run.get_outstanding_clarifications() == [unresolved_action]
    plan_run = portia.wait_for_ready(plan_run)
    assert plan_run.state == PlanRunState.READY_TO_RESUME
    assert plan_run.get_outstanding_clarifications() == []
    for clarification in plan_run.outputs.clarifications:
        if clarification.step == 1:
            assert clarification.resolved
            assert clarification.response == "complete"


def test_get_clarifications_and_get_run_called_once(portia: Portia) -> None:
    """Test that get_clarifications_for_step is called once after get_plan_run."""
    query = "example query"
    mock_response = StepsOrError(
        steps=[Step(task="Example task", inputs=[], output="$output")],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run(query)

    # Set the run state to NEED_CLARIFICATION to ensure it goes through the wait logic
    plan_run.state = PlanRunState.NEED_CLARIFICATION
    plan_run.current_step_index = 0  # Set to a valid index

    # Mock the storage methods
    with (
        mock.patch.object(
            portia.storage,
            "get_plan_run",
            return_value=plan_run,
        ) as mock_get_plan_run,
        mock.patch.object(
            PlanRun,
            "get_clarifications_for_step",
            return_value=[],
        ) as mock_get_clarifications,
    ):
        # Call wait_for_ready
        portia.wait_for_ready(plan_run)

        # Assert that get_run was called once
        mock_get_plan_run.assert_called_once_with(plan_run.id)

        # Assert that get_clarifications_for_step was called once after get_run
        mock_get_clarifications.assert_called_once()


def test_portia_run_query_with_summary(portia: Portia) -> None:
    """Test run_query sets both final output and summary correctly."""
    query = "What activities can I do in London based on weather?"

    # Mock planning_agent response
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
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan_run = portia.run(query)

        # Verify run completed successfully
        assert plan_run.state == PlanRunState.COMPLETE

        # Verify step outputs were stored correctly
        assert plan_run.outputs.step_outputs["$weather"] == weather_output
        assert plan_run.outputs.step_outputs["$activities"] == activities_output

        # Verify final output and summary
        assert plan_run.outputs.final_output is not None
        assert plan_run.outputs.final_output.value == activities_output.value
        assert plan_run.outputs.final_output.summary == expected_summary

        # Verify create_summary was called with correct args
        mock_summarizer_agent.create_summary.assert_called_once_with(
            plan=mock.ANY,
            plan_run=mock.ANY,
        )


def test_portia_sets_final_output_with_summary(portia: Portia) -> None:
    """Test that final output is set with correct summary."""
    (plan, plan_run) = get_test_plan_run()
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

    plan_run.outputs.step_outputs = {
        "$london_weather": Output(value="Sunny and warm"),
        "$activities": Output(value="Visit Hyde Park and have a picnic"),
    }

    expected_summary = "Weather is sunny and warm in London, visit to Hyde Park for a picnic"
    mock_summarizer = mock.MagicMock()
    mock_summarizer.create_summary.side_effect = [expected_summary]

    with mock.patch(
        "portia.portia.FinalOutputSummarizer",
        return_value=mock_summarizer,
    ):
        last_step_output = Output(value="Visit Hyde Park and have a picnic")
        output = portia._get_final_output(plan, plan_run, last_step_output)  # noqa: SLF001

        # Verify the final output
        assert output is not None
        assert output.value == "Visit Hyde Park and have a picnic"
        assert output.summary == expected_summary

        # Verify create_summary was called with correct args
        mock_summarizer.create_summary.assert_called_once()
        call_args = mock_summarizer.create_summary.call_args[1]
        assert isinstance(call_args["plan"], ReadOnlyPlan)
        assert isinstance(call_args["plan_run"], ReadOnlyPlanRun)
        assert call_args["plan"].id == plan.id
        assert call_args["plan_run"].id == plan_run.id


def test_portia_get_final_output_handles_summary_error(portia: Portia) -> None:
    """Test that final output is set even if summary generation fails."""
    (plan, plan_run) = get_test_plan_run()

    # Mock the SummarizerAgent to raise an exception
    mock_agent = mock.MagicMock()
    mock_agent.create_summary.side_effect = Exception("Summary failed")

    with mock.patch(
        "portia.execution_agents.utils.final_output_summarizer.FinalOutputSummarizer",
        return_value=mock_agent,
    ):
        step_output = Output(value="Some output")
        final_output = portia._get_final_output(plan, plan_run, step_output)  # noqa: SLF001

        # Verify the final output is set without summary
        assert final_output is not None
        assert final_output.value == "Some output"
        assert final_output.summary is None


def test_portia_wait_for_ready_max_retries(portia: Portia) -> None:
    """Test wait for ready with max retries."""
    plan, plan_run = get_test_plan_run()
    plan_run.state = PlanRunState.NEED_CLARIFICATION
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)
    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run, max_retries=0)


def test_portia_wait_for_ready_backoff_period(portia: Portia) -> None:
    """Test wait for ready with backoff period."""
    plan, plan_run = get_test_plan_run()
    plan_run.state = PlanRunState.NEED_CLARIFICATION
    portia.storage.save_plan(plan)
    portia.storage.get_plan_run = mock.MagicMock(return_value=plan_run)
    with pytest.raises(InvalidPlanRunStateError):
        portia.wait_for_ready(plan_run, max_retries=1, backoff_start_time_seconds=0)


def test_portia_resolve_clarification_error(portia: Portia) -> None:
    """Test resolve error."""
    plan, plan_run = get_test_plan_run()
    plan2, plan_run2 = get_test_plan_run()
    clarification = InputClarification(
        user_guidance="",
        argument_name="",
        plan_run_id=plan_run2.id,
    )
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)
    portia.storage.save_plan(plan2)
    portia.storage.save_plan_run(plan_run2)
    with pytest.raises(InvalidPlanRunStateError):
        portia.resolve_clarification(clarification, "test")

    with pytest.raises(InvalidPlanRunStateError):
        portia.resolve_clarification(clarification, "test", plan_run)


def test_portia_resolve_clarification(portia: Portia) -> None:
    """Test resolve success."""
    plan, plan_run = get_test_plan_run()
    clarification = InputClarification(
        user_guidance="",
        argument_name="",
        plan_run_id=plan_run.id,
    )
    plan_run.outputs.clarifications = [clarification]
    portia.storage.save_plan(plan)
    portia.storage.save_plan_run(plan_run)

    plan_run = portia.resolve_clarification(clarification, "test", plan_run)
    assert plan_run.state == PlanRunState.READY_TO_RESUME


def test_portia_get_tool_for_step_none_tool_id() -> None:
    """Test that when step.tool_id is None, LLMTool is used as fallback."""
    portia = Portia(config=get_test_config(), tools=[AdditionTool()])
    plan, plan_run = get_test_plan_run()

    # Create a step with no tool_id
    step = Step(
        task="Some task",
        inputs=[],
        output="$output",
        tool_id=None,
    )

    tool = portia._get_tool_for_step(step, plan_run)  # noqa: SLF001
    assert tool is None


def test_get_llm_tool() -> None:
    """Test special case retrieval of LLMTool as it isn't explicitly in most tool registries."""
    portia = Portia(config=get_test_config(), tools=example_tool_registry)
    plan, plan_run = get_test_plan_run()

    # Create a step with no tool_id
    step = Step(
        task="Some task",
        inputs=[],
        output="$output",
        tool_id=LLMTool.LLM_TOOL_ID,
    )

    tool = portia._get_tool_for_step(step, plan_run)  # noqa: SLF001
    assert tool is not None
    assert isinstance(tool._child_tool, LLMTool)  # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]


def test_portia_run_plan(portia: Portia) -> None:
    """Test that run_plan calls _create_plan_run and resume."""
    query = "example query"

    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan = portia.plan(query)

    # Mock the _create_plan_run and resume methods
    with (
        mock.patch.object(portia, "_create_plan_run") as mock_create_plan_run,
        mock.patch.object(portia, "resume") as mock_resume,
    ):
        mock_plan_run = MagicMock()
        mock_resumed_plan_run = MagicMock()
        mock_create_plan_run.return_value = mock_plan_run
        mock_resume.return_value = mock_resumed_plan_run

        result = portia.run_plan(plan)

        mock_create_plan_run.assert_called_once_with(plan)

        mock_resume.assert_called_once_with(mock_plan_run)

        assert result == mock_resumed_plan_run


def test_portia_handle_clarification() -> None:
    """Test that portia can handle a clarification."""
    clarification_handler = TestClarificationHandler()
    portia = Portia(
        config=get_test_config(),
        tools=[ClarificationTool()],
        execution_hooks=ExecutionHooks(clarification_handler=clarification_handler),
    )
    mock_plan = StepsOrError(
        steps=[
            Step(
                task="Raise a clarification",
                tool_id="clarification_tool",
                output="$output",
            ),
        ],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_plan)
    mock_step_agent = mock.MagicMock()
    mock_summarizer_agent = mock.MagicMock()
    mock_summarizer_agent.create_summary.side_effect = "I caught the clarification"
    with (
        mock.patch(
            "portia.portia.FinalOutputSummarizer",
            return_value=mock_summarizer_agent,
        ),
        mock.patch.object(portia, "_get_agent_for_step", return_value=mock_step_agent),
    ):
        plan = portia.plan("Raise a clarification")
        plan_run = portia._create_plan_run(plan)  # noqa: SLF001

        mock_step_agent.execute_sync.side_effect = [
            Output(
                value=InputClarification(
                    plan_run_id=plan_run.id,
                    user_guidance="Handle this clarification",
                    argument_name="raise_clarification",
                ),
            ),
            Output(value="I caught the clarification"),
        ]
        portia.resume(plan_run)
        assert plan_run.state == PlanRunState.COMPLETE

        # Check that the clarifications were handled correctly
        assert clarification_handler.received_clarification is not None
        assert (
            clarification_handler.received_clarification.user_guidance
            == "Handle this clarification"
        )


def test_portia_error_clarification(portia: Portia) -> None:
    """Test that portia can handle an error clarification."""
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run("test query")

    portia.error_clarification(
        ValueConfirmationClarification(
            plan_run_id=plan_run.id,
            user_guidance="Handle this clarification",
            argument_name="raise_clarification",
        ),
        error=ValueError("test error"),
    )
    assert plan_run.state == PlanRunState.FAILED


def test_portia_error_clarification_with_plan_run(portia: Portia) -> None:
    """Test that portia can handle an error clarification."""
    mock_response = StepsOrError(
        steps=[],
        error=None,
    )
    LLMWrapper.to_instructor = MagicMock(return_value=mock_response)

    plan_run = portia.run("test query")

    portia.error_clarification(
        ValueConfirmationClarification(
            plan_run_id=plan_run.id,
            user_guidance="Handle this clarification",
            argument_name="raise_clarification",
        ),
        error=ValueError("test error"),
        plan_run=plan_run,
    )
    assert plan_run.state == PlanRunState.FAILED
