"""E2E Tests."""

import pytest

from portia.config import AgentType, LLMProvider, default_config
from portia.plan import Plan, Step, Variable
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import WorkflowState
from tests.utils import AdditionTool, ClarificationTool, ErrorTool

PROVIDER_MODELS = [
    (
        LLMProvider.OPENAI,
        "gpt-4o-mini",
    ),
    (
        LLMProvider.MISTRALAI,
        "mistral-large-latest",
    ),
    (
        LLMProvider.ANTHROPIC,
        "claude-3-opus-20240229",
    ),
]
AGENTS = [
    AgentType.VERIFIER,
    AgentType.ONE_SHOT,
]


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_run_query(llm_provider: LLMProvider, llm_model_name: str, agent: AgentType) -> None:
    """Test running a simple query using the Runner."""
    config = default_config()
    config.llm_provider = llm_provider
    config.llm_model_name = llm_model_name
    config.default_agent_type = agent

    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    query = "Add 1 + 2 together"

    workflow = runner.run_query(query)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.final_output
    assert workflow.final_output.value == 3


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_plan_query(
    llm_provider: LLMProvider,
    llm_model_name: str,
    agent: AgentType,
) -> None:
    """Test planning a simple query using the Runner."""
    config = default_config()
    config.llm_provider = llm_provider
    config.llm_model_name = llm_model_name
    config.default_agent_type = agent

    tool_registry = InMemoryToolRegistry.from_local_tools([AdditionTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    query = "Add 1 + 2 together"

    plan = runner.plan_query(query)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "Add Tool"
    assert plan.steps[0].input
    assert len(plan.steps[0].input) == 2
    assert plan.steps[0].input[0].value + plan.steps[0].input[1].value == 3

    workflow = runner.run_plan(plan)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.final_output
    assert workflow.final_output.value == 3


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_run_query_with_clarifications(
    llm_provider: LLMProvider,
    llm_model_name: str,
    agent: AgentType,
) -> None:
    """Test running a query with clarification using the Runner."""
    config = default_config()
    config.llm_provider = llm_provider
    config.llm_model_name = llm_model_name
    config.default_agent_type = agent

    tool_registry = InMemoryToolRegistry.from_local_tools([ClarificationTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    clarification_step = Step(
        tool_name="Clarification Tool",
        task="Use tool",
        output="",
        input=[
            Variable(
                name="user_guidance",
                description="",
                value="Do Something",
            ),
            Variable(
                name="raise_clarification",
                description="",
                value=True,
            ),
        ],
    )
    plan = Plan(query="raise a clarification", steps=[clarification_step])
    runner.storage.save_plan(plan)
    workflow = runner.run_plan(plan)
    assert workflow.state == WorkflowState.NEED_CLARIFICATION
    assert workflow.get_outstanding_clarifications()[0].user_guidance == "Do Something"

    workflow.get_outstanding_clarifications()[0].resolve(response=False)
    runner.resume_workflow(workflow)
    assert workflow.state == WorkflowState.COMPLETE


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_run_query_with_error(
    llm_provider: LLMProvider,
    llm_model_name: str,
    agent: AgentType,
) -> None:
    """Test running a query with error using the Runner."""
    config = default_config()
    config.llm_provider = llm_provider
    config.llm_model_name = llm_model_name
    config.default_agent_type = agent

    tool_registry = InMemoryToolRegistry.from_local_tools([ErrorTool()])
    runner = Runner(config=config, tool_registry=tool_registry)
    clarification_step = Step(
        tool_name="Error Tool",
        task="Use tool",
        output="",
        input=[
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
        ],
    )
    plan = Plan(query="raise an error", steps=[clarification_step])
    workflow = runner.run_plan(plan)

    assert workflow.state == WorkflowState.FAILED
    assert workflow.final_output
    assert isinstance(workflow.final_output.value, str)
    assert "Something went wrong" in workflow.final_output.value
