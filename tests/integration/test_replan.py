"""E2E Tests."""

from __future__ import annotations

import pytest

from portia.config import AgentType, Config, LLMModel, LLMProvider, ReplanningMode
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, Step, Variable
from portia.runner import Runner
from portia.tool_registry import InMemoryToolRegistry
from portia.workflow import WorkflowState
from tests.utils import ErrorTool

PROVIDER_MODELS = [
    (
        LLMProvider.OPENAI,
        LLMModel.GPT_4_O_MINI,
    ),
]

AGENTS = [
    AgentType.VERIFIER,
]


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_replan_after_error_auto(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a query with error using the Runner."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
        replanning_mode=ReplanningMode.AUTOMATIC,
    )
    tool_registry = InMemoryToolRegistry.from_local_tools([ErrorTool(), LLMTool()])
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
                name="return_hard_error",
                description="",
                value=True,
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
            query="please call the error_tool and do not return an error by setting all the args to false.",
            tool_ids=["error_tool"],
        ),
        steps=[clarification_step],
    )
    runner.storage.save_plan(plan)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)
    print(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id != plan.id


@pytest.mark.parametrize(("llm_provider", "llm_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
def test_runner_replan_after_error_clarification(
    llm_provider: LLMProvider,
    llm_model_name: LLMModel,
    agent: AgentType,
) -> None:
    """Test running a query with error using the Runner."""
    config = Config.from_default(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        default_agent_type=agent,
        replanning_mode=ReplanningMode.CLARIFICATION,
    )
    tool_registry = InMemoryToolRegistry.from_local_tools([ErrorTool(), LLMTool()])
    runner = Runner(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="error_tool",
        task="Use tool",
        output="",
        inputs=[
            Variable(
                name="error_str",
                description="",
                value="the return_hard_error argument is true.",
            ),
            Variable(
                name="return_soft_error",
                description="",
                value=False,
            ),
            Variable(
                name="return_hard_error",
                description="",
                value=True,
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
            query="please call the error_tool and do not return an error by setting all the args to false.",
            tool_ids=["error_tool"],
        ),
        steps=[clarification_step],
    )
    runner.storage.save_plan(plan)
    workflow = runner.create_workflow(plan)
    workflow = runner.execute_workflow(workflow)
    assert workflow.state == WorkflowState.NEED_CLARIFICATION

    workflow = runner.resolve_clarification(
        workflow.get_outstanding_clarifications()[0],
        "yes",
        workflow,
    )

    runner.execute_workflow(workflow)

    assert workflow.state == WorkflowState.COMPLETE
    assert workflow.plan_id != plan.id
