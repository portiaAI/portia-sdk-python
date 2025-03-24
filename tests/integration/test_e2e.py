"""E2E Tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from unittest.mock import MagicMock, patch

import pytest
from pydantic import HttpUrl

from portia.clarification import ActionClarification, Clarification, InputClarification
from portia.clarification_handler import ClarificationHandler
from portia.config import Config, ExecutionAgentType, LLMModel, LLMProvider, LogLevel, StorageClass
from portia.errors import PlanError, ToolSoftError
from portia.open_source_tools.registry import example_tool_registry
from portia.plan import Plan, PlanContext, Step, Variable
from portia.plan_run import PlanRunState
from portia.portia import ExecutionHooks, Portia
from portia.tool_registry import InMemoryToolRegistry
from tests.utils import AdditionTool, ClarificationTool, ErrorTool, TestClarificationHandler

if TYPE_CHECKING:
    from portia.tool import ToolRunContext

PROVIDER_MODELS = [
    (
        LLMProvider.OPENAI,
        LLMModel.GPT_4_O_MINI,
    ),
    (
        LLMProvider.MISTRALAI,
        LLMModel.MISTRAL_LARGE,
    ),
    (
        LLMProvider.ANTHROPIC,
        LLMModel.CLAUDE_3_OPUS,
    ),
    (
        LLMProvider.GOOGLE_GENERATIVE_AI,
        LLMModel.GEMINI_2_0_FLASH,
    ),
]

AGENTS = [
    ExecutionAgentType.DEFAULT,
    ExecutionAgentType.ONE_SHOT,
]

@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_portia_run_query(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: ExecutionAgentType,
) -> None:
    """Test running a simple query."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    addition_tool = AdditionTool()
    addition_tool.should_summarize = True

    tool_registry = InMemoryToolRegistry.from_local_tools([addition_tool])
    portia = Portia(config=config, tools=tool_registry)
    query = "Add 1 + 2 together"

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    assert plan_run.outputs.final_output.value == 3
    for output in plan_run.outputs.step_outputs.values():
        assert output.summary is not None


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.flaky(reruns=3)
def test_portia_generate_plan(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: ExecutionAgentType,
) -> None:
    """Test planning a simple query."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    portia = Portia(config=config, tools=tool_registry)
    query = "Add 1 + 2 together"

    plan = portia.plan(query)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool_id == "add_tool"
    assert plan.steps[0].inputs
    assert len(plan.steps[0].inputs) == 2
    assert plan.steps[0].inputs[0].value + plan.steps[0].inputs[1].value == 3

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    assert plan_run.outputs.final_output.value == 3
    assert plan_run.outputs.final_output.summary is not None


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_portia_run_query_with_clarifications(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with clarification."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    test_clarification_handler = TestClarificationHandler()
    tool_registry = InMemoryToolRegistry.from_local_tools([ClarificationTool()])
    portia = Portia(
        config=config,
        tools=tool_registry,
        execution_hooks=ExecutionHooks(clarification_handler=test_clarification_handler),
    )
    clarification_step = Step(
        tool_id="clarification_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="user_guidance",
                description="",
                value="Return a clarification",
            ),
        ],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="raise a clarification",
            tool_ids=["clarification_tool"],
        ),
        steps=[clarification_step],
    )
    portia.storage.save_plan(plan)

    plan_run = portia.run_plan(plan)
    assert plan_run.state == PlanRunState.COMPLETE
    assert test_clarification_handler.received_clarification is not None
    assert (
        test_clarification_handler.received_clarification.user_guidance == "Return a clarification"
    )


def test_portia_run_query_with_clarifications_no_handler() -> None:
    """Test running a query with clarification using Portia."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=LLMProvider.OPENAI,
        llm_model_name=LLMModel.GPT_4_O_MINI,
        execution_agent_type=ExecutionAgentType.DEFAULT,
        storage_class=StorageClass.MEMORY,
    )

    tool_registry = InMemoryToolRegistry.from_local_tools([ClarificationTool()])
    portia = Portia(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="clarification_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="user_guidance",
                description="",
                value="Return a clarification",
            ),
        ],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="raise a clarification",
            tool_ids=["clarification_tool"],
        ),
        steps=[clarification_step],
    )
    portia.storage.save_plan(plan)

    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.NEED_CLARIFICATION
    assert plan_run.get_outstanding_clarifications()[0].user_guidance == "Return a clarification"

    plan_run = portia.resolve_clarification(
        plan_run.get_outstanding_clarifications()[0],
        "False",
    )

    portia.resume(plan_run)
    assert plan_run.state == PlanRunState.COMPLETE


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_portia_run_query_with_hard_error(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with error."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )
    tool_registry = InMemoryToolRegistry.from_local_tools([ErrorTool()])
    portia = Portia(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="error_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="error_str",
                description="",
                value="Something went wrong",
            ),
            Variable(
                name="return_soft_error",
                description="",
                value=False,
            ),
            Variable(
                name="return_uncaught_error",
                description="",
                value=False,
            ),
        ],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="raise an error",
            tool_ids=["error_tool"],
        ),
        steps=[clarification_step],
    )
    portia.storage.save_plan(plan)
    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.FAILED
    assert plan_run.outputs.final_output
    assert isinstance(plan_run.outputs.final_output.value, str)
    assert "Something went wrong" in plan_run.outputs.final_output.value


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.flaky(reruns=3)
def test_portia_run_query_with_soft_error(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with error."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    class MyAdditionTool(AdditionTool):
        def run(self, _: ToolRunContext, a: int, b: int) -> int:  # noqa: ARG002
            raise ToolSoftError("Server Timeout")

    tool_registry = InMemoryToolRegistry.from_local_tools([MyAdditionTool()])
    portia = Portia(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="add_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="a",
                description="",
                value=1,
            ),
            Variable(
                name="b",
                description="",
                value=2,
            ),
        ],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="raise an error",
            tool_ids=["add_tool"],
        ),
        steps=[clarification_step],
    )
    portia.storage.save_plan(plan)
    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.FAILED
    assert plan_run.outputs.final_output
    assert isinstance(plan_run.outputs.final_output.value, str)
    assert "Tool add_tool failed after retries" in plan_run.outputs.final_output.value


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.flaky(reruns=3)
def test_portia_run_query_with_multiple_clarifications(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with multiple clarification."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    class MyAdditionTool(AdditionTool):
        def run(self, ctx: ToolRunContext, a: int, b: int) -> int | Clarification:  # type: ignore  # noqa: PGH003
            if a == 1:
                return InputClarification(
                    plan_run_id=ctx.plan_run_id,
                    argument_name="a",
                    user_guidance="please try again",
                )
            return a + b

    test_clarification_handler = TestClarificationHandler()
    test_clarification_handler.clarification_response = 456
    tool_registry = InMemoryToolRegistry.from_local_tools([MyAdditionTool()])
    portia = Portia(
        config=config,
        tools=tool_registry,
        execution_hooks=ExecutionHooks(clarification_handler=test_clarification_handler),
    )

    step_one = Step(
        tool_id="add_tool",
        task="Use tool",
        output="$step_one",
        inputs=[
            Variable(
                name="a",
                description="",
                value=1,
            ),
            Variable(
                name="b",
                description="",
                value=2,
            ),
        ],
    )
    step_two = Step(
        tool_id="add_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="a",
                description="",
                value="$step_one",
            ),
            Variable(
                name="b",
                description="",
                value=40,
            ),
        ],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="raise a clarification",
            tool_ids=["clarification_tool"],
        ),
        steps=[step_one, step_two],
    )
    portia.storage.save_plan(plan)

    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    # 498 = 456 (clarification for value a in step 1) + 2 (value b in step 1)
    #  + 40 (value b in step 2)
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.value == 498
    assert plan_run.outputs.final_output.summary is not None

    assert test_clarification_handler.received_clarification is not None
    assert test_clarification_handler.received_clarification.user_guidance == "please try again"


@patch("time.sleep")
def test_portia_run_query_with_multiple_async_clarifications(
    sleep_mock: MagicMock,
) -> None:
    """Test running a query with multiple clarification."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        storage_class=StorageClass.CLOUD,
    )

    resolved = False

    class MyAdditionTool(AdditionTool):
        def run(self, ctx: ToolRunContext, a: int, b: int) -> int | Clarification:  # type: ignore  # noqa: PGH003
            nonlocal resolved
            if not resolved:
                return ActionClarification(
                    plan_run_id=ctx.plan_run_id,
                    user_guidance="please try again",
                    action_url=HttpUrl("https://www.test.com"),
                )
            resolved = False
            return a + b

    class ActionClarificationHandler(ClarificationHandler):
        def handle_action_clarification(
            self,
            clarification: ActionClarification,
            on_resolution: Callable[[Clarification, object], None],
            on_error: Callable[[Clarification, object], None],  # noqa: ARG002
        ) -> None:
            self.received_clarification = clarification

            # Call on_resolution and set the tool to return correctly after 2 sleeps in the
            # wait_for_ready loop
            def on_sleep_called(_: float) -> None:
                nonlocal resolved
                if sleep_mock.call_count >= 2:
                    sleep_mock.reset_mock()
                    on_resolution(clarification, 1)
                    resolved = True

            sleep_mock.side_effect = on_sleep_called

    test_clarification_handler = ActionClarificationHandler()
    portia = Portia(
        config=config,
        tools=InMemoryToolRegistry.from_local_tools([MyAdditionTool()]),
        execution_hooks=ExecutionHooks(clarification_handler=test_clarification_handler),
    )

    step_one = Step(
        tool_id="add_tool",
        task="Use tool",
        output="$step_one",
        inputs=[
            Variable(
                name="a",
                description="",
                value=1,
            ),
            Variable(
                name="b",
                description="",
                value=2,
            ),
        ],
    )
    step_two = Step(
        tool_id="add_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="a",
                description="",
                value=1,
            ),
            Variable(
                name="b",
                description="",
                value="$step_one",
            ),
        ],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="raise a clarification",
            tool_ids=["clarification_tool"],
        ),
        steps=[step_one, step_two],
    )
    portia.storage.save_plan(plan)

    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    # 4 = 1 (value a in step 1) + 2 (value b in step 1) + 1 (value a in step 2)
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.value == 4
    assert plan_run.outputs.final_output.summary is not None

    assert test_clarification_handler.received_clarification is not None
    assert test_clarification_handler.received_clarification.user_guidance == "please try again"


def test_portia_run_query_with_conditional_steps() -> None:
    """Test running a query with conditional steps."""
    config = Config.from_default(storage_class=StorageClass.MEMORY)
    portia = Portia(config=config, tools=example_tool_registry)
    query = "If the sum of 5 and 6 is greater than 10, then sum 3 + 4, otherwise sum 1 + 2"

    plan_run = portia.run(query)
    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.value == 7


def test_portia_run_query_with_example_registry() -> None:
    """Test we can run a query using the example registry."""
    config = Config.from_default()

    portia = Portia(config=config, tools=example_tool_registry)
    query = "Add 1 + 2 together and then write a haiku about the answer"

    plan_run = portia.run(query)
    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_query_requiring_cloud_tools_not_authenticated() -> None:
    """Test that running a query requiring cloud tools fails but points user to sign up."""
    config = Config.from_default(portia_api_key=None, storage_class=StorageClass.MEMORY)

    portia = Portia(config=config)
    query = "Send an email to John Doe"

    with pytest.raises(PlanError) as e:
        portia.plan(query)
    assert "PORTIA_API_KEY is required to use Portia cloud tools." in str(e.value)
