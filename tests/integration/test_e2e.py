"""E2E Tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from portia.clarification import Clarification, InputClarification
from portia.config import AgentType, Config, LLMModel, LLMProvider, LogLevel
from portia.errors import ToolSoftError
from portia.open_source_tools.registry import example_tool_registry
from portia.plan import Plan, PlanContext, Step, Variable
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import WorkflowState
from tests.utils import AdditionTool, ClarificationTool, ErrorTool

if TYPE_CHECKING:
    from portia.tool import ToolRunContext

PROVIDER_MODELS = [
    (
        LLMProvider.OPENAI,
        LLMModel.GPT_4_O_MINI,
    ),
    (
        LLMProvider.MISTRALAI,
        LLMModel.MISTRAL_LARGE_LATEST,
    ),
    (
        LLMProvider.ANTHROPIC,
        LLMModel.CLAUDE_3_OPUS_LATEST,
    ),
]

AGENTS = [
    AgentType.VERIFIER,
    AgentType.ONE_SHOT,
]


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_run_query(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a simple query using the Runner."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
    )

    addition_tool = AdditionTool()
    addition_tool.should_summarize = True

    tool_registry = InMemoryToolRegistry.from_local_tools([addition_tool])
    runner = Runner(config=config, tools=tool_registry)
    query = "Add 1 + 2 together"

    workflow = runner.execute_query(query)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.outputs.final_output
    assert workflow.outputs.final_output.value == 3
    for output in workflow.outputs.step_outputs.values():
        assert output.summary is not None


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.flaky(reruns=3)
def test_runner_generate_plan(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test planning a simple query using the Runner."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
    )

    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tools=tool_registry)
    query = "Add 1 + 2 together"

    plan = runner.generate_plan(query)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool_id == "add_tool"
    assert plan.steps[0].inputs
    assert len(plan.steps[0].inputs) == 2
    assert plan.steps[0].inputs[0].value + plan.steps[0].inputs[1].value == 3

    workflow = runner.execute_query(query)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.outputs.final_output
    assert workflow.outputs.final_output.value == 3
    assert workflow.outputs.final_output.summary is not None


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_run_query_with_clarifications(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a query with clarification using the Runner."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
    )

    tool_registry = InMemoryToolRegistry.from_local_tools([ClarificationTool()])
    runner = Runner(config=config, tools=tool_registry)
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
    runner.storage.save_plan(plan)

    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.NEED_CLARIFICATION
    assert workflow.get_outstanding_clarifications()[0].user_guidance == "Return a clarification"

    workflow = runner.resolve_clarification(
        workflow.get_outstanding_clarifications()[0],
        "False",
    )

    runner.execute_workflow(workflow)
    assert workflow.state == WorkflowState.COMPLETE


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_run_query_with_hard_error(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a query with error using the Runner."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
    )
    tool_registry = InMemoryToolRegistry.from_local_tools([ErrorTool()])
    runner = Runner(config=config, tools=tool_registry)
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
    runner.storage.save_plan(plan)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.FAILED
    assert workflow.outputs.final_output
    assert isinstance(workflow.outputs.final_output.value, str)
    assert "Something went wrong" in workflow.outputs.final_output.value


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.flaky(reruns=3)
def test_runner_run_query_with_soft_error(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a query with error using the Runner."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
    )

    class MyAdditionTool(AdditionTool):
        def run(self, _: ToolRunContext, a: int, b: int) -> int:  # noqa: ARG002
            raise ToolSoftError("Server Timeout")

    tool_registry = InMemoryToolRegistry.from_local_tools([MyAdditionTool()])
    runner = Runner(config=config, tools=tool_registry)
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
    runner.storage.save_plan(plan)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.FAILED
    assert workflow.outputs.final_output
    assert isinstance(workflow.outputs.final_output.value, str)
    assert "Tool add_tool failed after retries" in workflow.outputs.final_output.value


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.flaky(reruns=3)
def test_runner_run_query_with_multiple_clarifications(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a query with multiple clarification using the Runner."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
    )

    class MyAdditionTool(AdditionTool):
        def run(self, ctx: ToolRunContext, a: int, b: int) -> int | Clarification:  # type: ignore  # noqa: PGH003
            if a == 1:
                return InputClarification(
                    workflow_id=ctx.workflow_id,
                    argument_name="a",
                    user_guidance="please try again",
                )
            return a + b

    tool_registry = InMemoryToolRegistry.from_local_tools([MyAdditionTool()])
    runner = Runner(config=config, tools=tool_registry)

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
    runner.storage.save_plan(plan)

    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.NEED_CLARIFICATION
    assert workflow.get_outstanding_clarifications()[0].user_guidance == "please try again"

    workflow = runner.resolve_clarification(
        workflow.get_outstanding_clarifications()[0],
        456,
        workflow,
    )

    runner.execute_workflow(workflow)
    assert workflow.state == WorkflowState.COMPLETE
    # 498 = 456 (clarification - value a - step 1) + 2 (value b - step 1) + 40 (value b - step 2)
    assert workflow.outputs.final_output is not None
    assert workflow.outputs.final_output.value == 498
    assert workflow.outputs.final_output.summary is not None


def test_runner_run_query_with_example_registry() -> None:
    """Test we can run a query using the example registry."""
    config = Config.from_default()

    runner = Runner(config=config, tools=example_tool_registry)
    query = "Add 1 + 2 together and then write a haiku about the answer"

    workflow = runner.execute_query(query)
    assert workflow.state == WorkflowState.COMPLETE
