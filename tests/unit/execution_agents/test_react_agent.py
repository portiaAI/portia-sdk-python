"""Test ReAct agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from portia.builder.plan_builder import PlanBuilderV2
from portia.clarification import InputClarification
from portia.errors import InvalidAgentError
from portia.execution_agents.output import LocalDataValue
from portia.execution_agents.react_agent import FinalResultTool, ReActAgent
from portia.execution_agents.react_clarification_tool import ReActClarificationTool
from portia.execution_hooks import ExecutionHooks
from portia.run_context import RunContext
from tests.utils import (
    AdditionTool,
    get_mock_generative_model,
    get_test_config,
    get_test_plan_run,
    get_test_tool_context,
)

if TYPE_CHECKING:
    from portia.tool import ToolRunContext


class OutputSchema(BaseModel):
    """Test structured output schema."""

    result: str = Field(..., description="The final result")
    count: int = Field(..., description="Number of operations performed")


class MultiplyTool(AdditionTool):
    """A multiplication tool for testing multiple tools."""

    id: str = "multiply_tool"
    name: str = "Multiply Tool"
    description: str = "Multiply two numbers together"

    def run(self, _: ToolRunContext, a: int, b: int) -> int:
        """Multiply the numbers."""
        return a * b

    async def arun(self, _: ToolRunContext, a: int, b: int) -> int:
        """Multiply the numbers asynchronously."""
        return a * b


@pytest.fixture
def mock_run_context() -> RunContext:
    """Create a mock run context for testing."""
    plan, plan_run = get_test_plan_run()
    config = get_test_config()

    # Create mock run context
    run_context = mock.MagicMock(spec=RunContext)
    run_context.legacy_plan = plan
    run_context.plan_run = plan_run
    run_context.config = config
    run_context.telemetry = mock.MagicMock()
    run_context.execution_hooks = ExecutionHooks()
    run_context.get_tool_run_ctx.return_value = get_test_tool_context(plan_run, config)

    return run_context


def test_react_agent_initialization(mock_run_context: RunContext) -> None:
    """Test ReActAgent can be properly initialized."""
    tools = [AdditionTool(), MultiplyTool()]
    agent = ReActAgent(
        task="Test task",
        task_data={"key": "value"},
        tools=tools,
        run_data=mock_run_context,
        tool_call_limit=20,
        allow_agent_clarifications=True,
    )

    assert agent.task == "Test task"
    assert agent.task_data == {"key": "value"}
    assert len(agent.tools) == 2
    assert agent.tool_call_limit == 20
    assert agent.allow_agent_clarifications is True
    assert agent.run_data == mock_run_context


@pytest.mark.asyncio
async def test_react_agent_immediate_final_result(mock_run_context: RunContext) -> None:
    """Test agent that calls final result tool immediately without any other tool calls."""
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Add numbers").build()
    mock_model_response = AIMessage(
        content="I can answer directly",
        tool_calls=[
            {
                "name": FinalResultTool().get_langchain_name(),
                "type": "tool_call",
                "id": "call_1",
                "args": {"final_result": "The answer is 42"},
            }
        ],
    )

    mock_agent_model = get_mock_generative_model(response=mock_model_response)

    mock_summarizer_model = get_mock_generative_model()
    mock_summarizer_model.get_response = MagicMock()
    mock_summarizer_model.get_response.return_value.content = "Direct answer provided"

    with (
        mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model),
        mock.patch("portia.config.Config.get_summarizer_model", return_value=mock_summarizer_model),
    ):
        tools = [AdditionTool()]
        agent = ReActAgent(
            task="What is the answer to life?",
            task_data=None,
            tools=tools,
            run_data=mock_run_context,
            tool_call_limit=10,
            allow_agent_clarifications=False,
        )

        result = await agent.execute()

        assert isinstance(result, LocalDataValue)
        assert result.value == "The answer is 42"
        assert result.summary == "Direct answer provided"


@pytest.mark.asyncio
async def test_react_agent_single_tool_call_then_final_result(mock_run_context: RunContext) -> None:
    """Test agent runs with single tool call, then calls final result tool - no structured."""
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Add numbers").build()
    mock_agent_model = get_mock_generative_model()
    mock_agent_model.to_langchain().ainvoke.side_effect = [  # pyright: ignore[reportAttributeAccessIssue]
        AIMessage(
            content="I'll add the numbers first",
            tool_calls=[
                {"name": "Add_Tool", "type": "tool_call", "id": "call_1", "args": {"a": 3, "b": 4}}
            ],
        ),
        AIMessage(
            content="Now I'll provide the final result",
            tool_calls=[
                {
                    "name": FinalResultTool().get_langchain_name(),
                    "type": "tool_call",
                    "id": "call_2",
                    "args": {"final_result": "The sum is 7"},
                }
            ],
        ),
    ]

    mock_summarizer_model = get_mock_generative_model()
    mock_summarizer_model.get_response = MagicMock()
    mock_summarizer_model.get_response.return_value.content = "Addition completed successfully"
    with (
        mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model),
        mock.patch("portia.config.Config.get_summarizer_model", return_value=mock_summarizer_model),
    ):
        tools = [AdditionTool(), MultiplyTool()]
        agent = ReActAgent(
            task="Add 3 and 4",
            task_data=["test123", 456789],
            tools=tools,
            run_data=mock_run_context,
        )

        result = await agent.execute()

        assert isinstance(result, LocalDataValue)
        assert result.value == "The sum is 7"
        assert result.summary == "Addition completed successfully"

        # Check that task_data is correctly passed through to the model
        for call in mock_agent_model.to_langchain().ainvoke.call_args_list:  # pyright: ignore[reportAttributeAccessIssue]
            assert "test123" in call[0][0][1].content
            assert "456789" in call[0][0][1].content


@pytest.mark.asyncio
async def test_react_agent_multiple_tool_calls_then_final_result(
    mock_run_context: RunContext,
) -> None:
    """Test agent runs with 3 tool calls to 2 different tools, then final result."""
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Add numbers").build()
    mock_agent_model = get_mock_generative_model()
    mock_agent_model.to_langchain().ainvoke.side_effect = [  # pyright: ignore[reportAttributeAccessIssue]
        AIMessage(
            content="First I'll add 2 and 3",
            tool_calls=[
                {"name": "Add_Tool", "type": "tool_call", "id": "call_1", "args": {"a": 2, "b": 3}}
            ],
        ),
        AIMessage(
            content="Now I'll multiply the result by 4",
            tool_calls=[
                {
                    "name": "Multiply_Tool",
                    "type": "tool_call",
                    "id": "call_2",
                    "args": {"a": 5, "b": 4},
                }
            ],
        ),
        AIMessage(
            content="Finally I'll add 10 to get the final answer",
            tool_calls=[
                {
                    "name": "Add_Tool",
                    "type": "tool_call",
                    "id": "call_3",
                    "args": {"a": 20, "b": 10},
                }
            ],
        ),
        AIMessage(
            content="Now I'll provide the final result",
            tool_calls=[
                {
                    "name": FinalResultTool().get_langchain_name(),
                    "type": "tool_call",
                    "id": "call_4",
                    "args": {"final_result": "Final answer: 30 after performing 3 operations"},
                }
            ],
        ),
    ]

    mock_summarizer_model = get_mock_generative_model()
    mock_summarizer_model.get_response = MagicMock()
    mock_summarizer_model.get_response.return_value.content = "Operations completed successfully"

    with (
        mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model),
        mock.patch("portia.config.Config.get_summarizer_model", return_value=mock_summarizer_model),
    ):
        tools = [AdditionTool(), MultiplyTool()]
        agent = ReActAgent(
            task="Perform mathematical operations: add 2+3, multiply by 4, then add 10",
            task_data=None,
            tools=tools,
            run_data=mock_run_context,
        )

        result = await agent.execute()

        assert isinstance(result, LocalDataValue)
        assert result.value == "Final answer: 30 after performing 3 operations"
        assert result.summary == "Operations completed successfully"


@pytest.mark.asyncio
async def test_react_agent_with_clarification_tool(mock_run_context: RunContext) -> None:
    """Test agent calls clarification tool and we receive clarification as output."""
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Add numbers").build()
    mock_agent_model = get_mock_generative_model()
    mock_agent_model.to_langchain().ainvoke.side_effect = [  # pyright: ignore[reportAttributeAccessIssue]
        AIMessage(
            content="I need clarification",
            tool_calls=[
                {
                    "name": ReActClarificationTool().get_langchain_name(),
                    "type": "tool_call",
                    "id": "call_1",
                    "args": {"guidance": "What specific operation should I perform?"},
                }
            ],
        ),
        AIMessage(
            content="I'll add the numbers",
            tool_calls=[
                {"name": "Add_Tool", "type": "tool_call", "id": "call_2", "args": {"a": 3, "b": 4}}
            ],
        ),
        AIMessage(
            content="Now I'll provide the final result",
            tool_calls=[
                {
                    "name": FinalResultTool().get_langchain_name(),
                    "type": "tool_call",
                    "id": "call_3",
                    "args": {"final_result": "The sum is 7"},
                }
            ],
        ),
    ]

    mock_summarizer_model = get_mock_generative_model()
    mock_summarizer_model.get_response = MagicMock()
    mock_summarizer_model.get_response.return_value.content = "Addition completed successfully"

    with (
        mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model),
        mock.patch("portia.config.Config.get_summarizer_model", return_value=mock_summarizer_model),
    ):
        tools = [AdditionTool()]
        agent = ReActAgent(
            task="Perform some operation",
            task_data=None,
            tools=tools,
            run_data=mock_run_context,
            allow_agent_clarifications=True,
        )

        result = await agent.execute()

        # Verify we got the clarification as output
        assert isinstance(result, LocalDataValue)
        assert isinstance(result.value, list)
        assert len(result.value) == 1
        assert isinstance(result.value[0], InputClarification)
        clarification = result.value[0]
        assert clarification.user_guidance == "What specific operation should I perform?"
        assert clarification.argument_name == "react_agent_clarification"
        assert clarification.source == "ReActClarificationTool"

        # Now handle the clarification, run again and check we continue successfully
        clarification.response = "You should add the numbers"
        clarification.resolved = True
        clarification.step = 0
        mock_run_context.plan_run.outputs.clarifications = [clarification]
        result = await agent.execute()
        assert isinstance(result, LocalDataValue)
        assert result.value == "The sum is 7"
        assert result.summary == "Addition completed successfully"

        # Verify that the second call to the model included the clarification response
        second_call_args = mock_agent_model.to_langchain().ainvoke.call_args_list[1]  # pyright: ignore[reportAttributeAccessIssue]
        assert "You should add the numbers" in second_call_args[0][0][1].content


@pytest.mark.asyncio
async def test_react_agent_recursion_limit_error(mock_run_context: RunContext) -> None:
    """Test that an error is thrown when recursion limit is hit."""
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Add numbers").build()
    call_counter = 0

    def generate_response(*_args: object, **_kwargs: object) -> AIMessage:
        nonlocal call_counter
        call_counter += 1
        return AIMessage(
            content="Keep adding",
            tool_calls=[
                {
                    "name": "Add_Tool",
                    "type": "tool_call",
                    "id": f"call_{call_counter}",
                    "args": {"a": 1, "b": 1},
                }
            ],
        )

    mock_agent_model = get_mock_generative_model()
    mock_agent_model.to_langchain().ainvoke.side_effect = generate_response  # type: ignore[reportAttributeAccessIssue]

    with mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model):
        tools = [AdditionTool()]
        agent = ReActAgent(
            task="Keep adding numbers",
            task_data=None,
            tools=tools,
            run_data=mock_run_context,
            tool_call_limit=5,  # Relatively low limit to trigger error
        )

        with pytest.raises(InvalidAgentError):
            await agent.execute()


@pytest.mark.asyncio
async def test_react_agent_execution_hooks_called(mock_run_context: RunContext) -> None:
    """Test that execution hooks are called correctly."""
    mock_run_context.execution_hooks = ExecutionHooks(
        before_tool_call=mock.MagicMock(return_value=None), after_tool_call=mock.MagicMock()
    )
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Complete task").build()

    mock_model_response = AIMessage(
        content="Task done",
        tool_calls=[
            {
                "name": "Final_Result_Tool",
                "type": "tool_call",
                "id": "call_1",
                "args": {"final_result": "Complete"},
            }
        ],
    )
    mock_agent_model = get_mock_generative_model(mock_model_response)

    mock_summarizer_model = get_mock_generative_model()
    mock_summarizer_model.get_response = MagicMock()
    mock_summarizer_model.get_response.return_value.content = "Task finished"

    with (
        mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model),
        mock.patch("portia.config.Config.get_summarizer_model", return_value=mock_summarizer_model),
    ):
        tools = [AdditionTool()]
        agent = ReActAgent(
            task="Complete task",
            task_data=None,
            tools=tools,
            run_data=mock_run_context,
        )

        result = await agent.execute()

        # Verify result
        assert isinstance(result, LocalDataValue)
        assert result.value == "Complete"
        assert result.summary == "Task finished"

        assert mock_run_context.execution_hooks.before_tool_call.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
        assert mock_run_context.execution_hooks.after_tool_call.call_count == 1  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_react_agent_prestep_clarification(mock_run_context: RunContext) -> None:
    """Test pre-step execution hook that throws a clarification."""
    mock_run_context.plan = PlanBuilderV2().react_agent_step(task="Add numbers").build()
    before_tool_call_mock = mock.MagicMock(
        return_value=InputClarification(
            plan_run_id=mock_run_context.plan_run.id,
            user_guidance="Need permission before proceeding",
            argument_name="permission",
            source="Pre-step hook",
        )
    )
    mock_run_context.execution_hooks = ExecutionHooks(before_tool_call=before_tool_call_mock)

    response = AIMessage(
        content="Calling tool",
        tool_calls=[
            {"name": "Add_Tool", "type": "tool_call", "id": "call_1", "args": {"a": 1, "b": 2}}
        ],
    )
    mock_agent_model = get_mock_generative_model(response)

    with mock.patch("portia.config.Config.get_planning_model", return_value=mock_agent_model):
        tools = [AdditionTool()]
        agent = ReActAgent(
            task="Add numbers",
            task_data=None,
            tools=tools,
            run_data=mock_run_context,
        )

        result = await agent.execute()
        assert isinstance(result, LocalDataValue)
        assert isinstance(result.value, InputClarification)
        assert result.value.user_guidance == "Need permission before proceeding"
        assert result.value.argument_name == "permission"
        assert result.value.source == "Pre-step hook"


@pytest.mark.asyncio
async def test_final_result_tool() -> None:
    """Test FinalResultTool functionality."""
    # Test basic functionality - FinalResultTool returns the final result as string
    tool = FinalResultTool()
    ctx = get_test_tool_context()

    result = tool.run(ctx, "The answer is 42")

    # FinalResultTool.run() returns the final result string directly
    assert isinstance(result, str)
    assert result == "The answer is 42"
