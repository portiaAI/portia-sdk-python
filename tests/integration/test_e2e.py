"""E2E Tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, HttpUrl

from portia.clarification import ActionClarification, Clarification, InputClarification
from portia.clarification_handler import ClarificationHandler
from portia.config import (
    Config,
    ExecutionAgentType,
    LogLevel,
    StorageClass,
)
from portia.errors import PlanError, ToolHardError, ToolSoftError
from portia.execution_agents.output import LocalDataValue
from portia.model import LLMProvider
from portia.open_source_tools.registry import example_tool_registry, open_source_tool_registry
from portia.plan import Plan, PlanBuilder, PlanContext, PlanInput, Step, Variable
from portia.plan_run import PlanRunState
from portia.portia import ExecutionHooks, Portia
from portia.tool import Tool
from portia.tool_registry import ToolRegistry
from tests.utils import AdditionTool, ClarificationTool, ErrorTool, TestClarificationHandler

if TYPE_CHECKING:
    from portia.tool import ToolRunContext


CORE_PROVIDERS = [
    (
        LLMProvider.OPENAI,
        "openai/gpt-4o-mini",
    ),
    (
        LLMProvider.ANTHROPIC,
        "anthropic/claude-3-5-sonnet-latest",
    ),
]

PLANNING_PROVIDERS = [
    (
        LLMProvider.OPENAI,
        "openai/o3-mini",
    ),
    (
        LLMProvider.ANTHROPIC,
        "anthropic/claude-3-5-sonnet-latest",
    ),
]

PROVIDER_MODELS = [
    *CORE_PROVIDERS,
    (
        LLMProvider.MISTRALAI,
        "mistralai/mistral-large-latest",
    ),
    (
        LLMProvider.GOOGLE_GENERATIVE_AI,
        "google/gemini-2.0-flash",
    ),
]

AGENTS = [
    ExecutionAgentType.DEFAULT,
    ExecutionAgentType.ONE_SHOT,
]


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PROVIDER_MODELS)
@pytest.mark.flaky(reruns=4)
def test_portia_run_query(
    llm_provider: LLMProvider,
    default_model_name: str,
) -> None:
    """Test running a simple query."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
        storage_class=StorageClass.MEMORY,
    )

    addition_tool = AdditionTool()
    addition_tool.should_summarize = True

    tool_registry = ToolRegistry([addition_tool])
    portia = Portia(config=config, tools=tool_registry)
    query = "Add 1 + 2"

    plan_run = portia.run(query)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output
    assert plan_run.outputs.final_output.get_value() == 3
    for output in plan_run.outputs.step_outputs.values():
        assert output.get_summary() is not None


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PROVIDER_MODELS)
@pytest.mark.flaky(reruns=4)
def test_portia_generate_plan(
    llm_provider: LLMProvider,
    default_model_name: str,
) -> None:
    """Test planning a simple query."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
        storage_class=StorageClass.MEMORY,
    )

    tool_registry = ToolRegistry([AdditionTool()])
    portia = Portia(config=config, tools=tool_registry)
    query = "Add 1 + 2"

    plan = portia.plan(query)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool_id == "add_tool"


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PROVIDER_MODELS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.flaky(reruns=3)
def test_portia_run_query_with_clarifications(
    llm_provider: LLMProvider,
    default_model_name: str,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with clarification."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=llm_provider,
        default_model=default_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    test_clarification_handler = TestClarificationHandler()
    tool_registry = ToolRegistry([ClarificationTool()])
    portia = Portia(
        config=config,
        tools=tool_registry,
        execution_hooks=ExecutionHooks(clarification_handler=test_clarification_handler),
    )
    clarification_step = Step(
        tool_id="clarification_tool",
        task="Raise a clarification with user guidance 'Return a clarification'",
        output="",
        inputs=[],
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
        default_model="openai/gpt-4o-mini",
        execution_agent_type=ExecutionAgentType.DEFAULT,
        storage_class=StorageClass.MEMORY,
    )

    tool_registry = ToolRegistry([ClarificationTool()])
    portia = Portia(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="clarification_tool",
        task="raise a clarification with a user guidance 'Return a clarification'",
        output="",
        inputs=[],
    )
    plan = Plan(
        plan_context=PlanContext(
            query="Raise a clarification",
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


@pytest.mark.parametrize(("llm_provider", "default_model_name"), CORE_PROVIDERS)
@pytest.mark.parametrize("agent", AGENTS)
def test_portia_run_query_with_hard_error(
    llm_provider: LLMProvider,
    default_model_name: str,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with error."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )
    tool_registry = ToolRegistry([ErrorTool()])
    portia = Portia(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="error_tool",
        task="Use error tool with string 'Something went wrong' and \
        do not return a soft error or uncaught error",
        output="",
        inputs=[],
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
    final_output = plan_run.outputs.final_output.get_value()
    assert isinstance(final_output, str)
    assert "Something went wrong" in final_output


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize(("llm_provider", "default_model_name"), CORE_PROVIDERS)
@pytest.mark.flaky(reruns=3)
def test_portia_run_query_with_soft_error(
    llm_provider: LLMProvider,
    default_model_name: str,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with error."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    class MyAdditionTool(AdditionTool):
        def run(self, _: ToolRunContext, a: int, b: int) -> int:  # noqa: ARG002
            raise ToolSoftError("Server Timeout")

    tool_registry = ToolRegistry([MyAdditionTool()])
    portia = Portia(config=config, tools=tool_registry)
    clarification_step = Step(
        tool_id="add_tool",
        task="Add 1 + 2",
        output="",
        inputs=[],
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
    final_output = plan_run.outputs.final_output.get_value()
    assert isinstance(final_output, str)
    assert "Tool add_tool failed after retries" in final_output


@pytest.mark.parametrize(("llm_provider", "default_model_name"), CORE_PROVIDERS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.flaky(reruns=3)
def test_portia_run_query_with_multiple_clarifications(
    llm_provider: LLMProvider,
    default_model_name: str,
    agent: ExecutionAgentType,
) -> None:
    """Test running a query with multiple clarification."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        llm_provider=llm_provider,
        default_model=default_model_name,
        execution_agent_type=agent,
        storage_class=StorageClass.MEMORY,
    )

    class MyAdditionTool(AdditionTool):
        retries: int = 0

        def run(self, ctx: ToolRunContext, a: int, b: int) -> int | Clarification:  # type: ignore  # noqa: PGH003
            # Avoid an endless loop of clarifications
            if self.retries > 2:
                raise ToolHardError("Tool failed after 2 retries")
            if a == 1:
                self.retries += 1
                return InputClarification(
                    plan_run_id=ctx.plan_run_id,
                    argument_name="a",
                    user_guidance="please try again",
                )
            return a + b

    test_clarification_handler = TestClarificationHandler()
    test_clarification_handler.clarification_response = 456
    tool_registry = ToolRegistry([MyAdditionTool()])
    portia = Portia(
        config=config,
        tools=tool_registry,
        execution_hooks=ExecutionHooks(clarification_handler=test_clarification_handler),
    )

    step_one = Step(
        tool_id="add_tool",
        task="Add 1 + 2",
        output="$step_one",
        inputs=[],
    )
    step_two = Step(
        tool_id="add_tool",
        task="Add $step_one + 40",
        output="",
        inputs=[
            Variable(
                name="$step_one",
                description="value for step one",
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
    assert plan_run.outputs.final_output.get_value() == 498
    assert plan_run.outputs.final_output.get_summary() is not None

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
        tools=ToolRegistry([MyAdditionTool()]),
        execution_hooks=ExecutionHooks(clarification_handler=test_clarification_handler),
    )

    step_one = Step(
        tool_id="add_tool",
        task="Add 1 + 2",
        output="$step_one",
        inputs=[],
    )
    step_two = Step(
        tool_id="add_tool",
        task="Add $step_one + 1",
        output="",
        inputs=[
            Variable(
                name="$step_one",
                description="value for step one",
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
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == 4
    assert plan_run.outputs.final_output.get_summary() is not None

    assert test_clarification_handler.received_clarification is not None
    assert test_clarification_handler.received_clarification.user_guidance == "please try again"


@pytest.mark.flaky(reruns=3)
def test_portia_run_query_with_conditional_steps() -> None:
    """Test running a query with conditional steps."""
    config = Config.from_default(storage_class=StorageClass.MEMORY)
    portia = Portia(config=config, tools=example_tool_registry)
    query = (
        "If the sum of 5 and 6 is greater than 10, then sum 4 + 5 and give me the answer, "
        "otherwise sum 1 + 2 and give me that as the answer"
    )

    plan_run = portia.run(query)
    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None
    assert "9" in str(plan_run.outputs.final_output.get_value())
    assert "3" not in str(plan_run.outputs.final_output.get_value())


def test_portia_run_query_with_example_registry() -> None:
    """Test we can run a query using the example registry."""
    config = Config.from_default()

    portia = Portia(config=config, tools=open_source_tool_registry)
    query = """Add 1 + 2 together and then write me a haiku about the answer.
    You can use the LLM tool to generate a haiku."""

    plan_run = portia.run(query)
    assert plan_run.state == PlanRunState.COMPLETE


def test_portia_run_query_requiring_cloud_tools_not_authenticated() -> None:
    """Test that running a query requiring cloud tools fails but points user to sign up."""
    config = Config.from_default(
        portia_api_key=None,
        storage_class=StorageClass.MEMORY,
        default_log_level=LogLevel.DEBUG,
    )

    portia = Portia(config=config)
    query = (
        "Send an email to John Doe (john.doe@example.com) using the Gmail tool. Only use the Gmail "
        "tool and fail if you can't use it."
    )

    with pytest.raises(PlanError) as e:
        portia.plan(query)
    assert "PORTIA_API_KEY is required to use Portia cloud tools." in str(e.value)


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PLANNING_PROVIDERS)
def test_portia_plan_steps_inputs_dependencies(
    llm_provider: LLMProvider,
    default_model_name: str,
) -> None:
    """Test that a dynamically generated plan properly creates step dependencies."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
        storage_class=StorageClass.MEMORY,
    )

    portia = Portia(config=config, tools=open_source_tool_registry)

    query = """First calculate 25 * 3, then write a haiku about the result,
    and finally summarize the haiku with the result of the calculation.
    If the result of the calculation is greater than 100, then final summary should
     be saved to a file.
    """

    plan = portia.plan(query)

    assert len(plan.steps) == 4, "Plan should have 4 steps"

    assert plan.steps[0].inputs == [], "First step should not have inputs"
    assert plan.steps[0].tool_id == "calculator_tool", "First step should have the calculator tool"
    assert plan.steps[0].condition is None, "First step should not have a condition"

    assert len(plan.steps[1].inputs) == 1, "Second step should have 1 input from calculation step"
    assert (
        plan.steps[1].inputs[0].name == plan.steps[0].output
    ), "Second step should equal the output of the first step"
    assert plan.steps[1].tool_id == "llm_tool", "Second step should have the LLM tool"
    assert plan.steps[1].condition is None, "Second step should not have a condition"

    assert len(plan.steps[2].inputs) == 2, "Third step should have (calculation and haiku) inputs"
    assert {inp.name for inp in plan.steps[2].inputs} == {
        plan.steps[0].output,
        plan.steps[1].output,
    }, "Third step inputs should match outputs of (calculation and haiku) steps"
    assert plan.steps[2].condition is None, "Third step should not have a condition"
    assert plan.steps[2].tool_id == "llm_tool", "Third step should be llm_tool"

    assert len(plan.steps[3].inputs) >= 1, "Fourth step should have summary input"
    assert any(
        inp.name == plan.steps[2].output for inp in plan.steps[3].inputs
    ), "Fourth step inputs should have summary input"
    assert plan.steps[3].tool_id == "file_writer_tool", "Fourth step should be file_writer_tool"
    assert "100" in str(plan.steps[3].condition), "Fourth step condition does not contain 100"


def test_plan_input_with_schema_validation() -> None:
    """Test running a plan with schema-validated plan inputs."""

    class AdditionNumbers(BaseModel):
        num_a: int = Field(description="First number to add")
        num_b: int = Field(description="Second number to add")

    numbers_input = PlanInput(name="$numbers", description="two numbers to add together")

    plan_inputs = {numbers_input: LocalDataValue(value=AdditionNumbers(num_a=5, num_b=7))}

    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        storage_class=StorageClass.MEMORY,
    )
    portia = Portia(config=config, tools=ToolRegistry([AdditionTool()]))
    plan = portia.plan(
        "Use the addition tool to add together the two provided numbers",
        plan_inputs=[numbers_input],
    )
    plan_run = portia.run_plan(plan, plan_run_inputs=plan_inputs)

    assert plan_run.state == PlanRunState.COMPLETE
    assert plan_run.outputs.final_output is not None
    assert plan_run.outputs.final_output.get_value() == 12  # 5 + 7 = 12

    # Check that plan inputs were stored correctly
    assert "$numbers" in plan_run.plan_run_inputs
    assert plan_run.plan_run_inputs["$numbers"].get_value().num_a == 5  # pyright: ignore[reportOptionalMemberAccess]
    assert plan_run.plan_run_inputs["$numbers"].get_value().num_b == 7  # pyright: ignore[reportOptionalMemberAccess]


def test_run_plan_with_large_step_input() -> None:
    """Test running a plan with a large step input."""
    config = Config.from_default(
        default_log_level=LogLevel.DEBUG,
        storage_class=StorageClass.MEMORY,
        llm_provider=LLMProvider.ANTHROPIC,
    )

    class StoryToolSchema(BaseModel):
        """Input for StoryTool."""

    class StoryTool(Tool):
        """A tool that returns the first chapter of War and Peace."""

        id: str = "story_tool"
        name: str = "Story Tool"
        description: str = "Returns the first chapter of War and Peace"
        args_schema: type[BaseModel] = StoryToolSchema
        output_schema: tuple[str, str] = ("str", "str: The first chapter of War and Peace")

        def run(self, _: ToolRunContext) -> str:
            """Return the first chapter of War and Peace."""
            path = Path(__file__).parent / "data" / "war_and_peace_ch1.txt"
            with path.open() as f:
                return f.read()

    class EmailToolSchema(BaseModel):
        """Input for EmailTool."""

        recipient: str = Field(..., description="The email address of the recipient")
        subject: str = Field(..., description="The subject line of the email")
        body: str = Field(..., description="The content of the email")

    email_tool_called = False

    class EmailTool(Tool):
        """A tool that mocks sending an email."""

        id: str = "email_tool"
        name: str = "Email Tool"
        description: str = "Sends an email to a recipient"
        args_schema: type[BaseModel] = EmailToolSchema
        output_schema: tuple[str, str] = ("str", "str: Confirmation message for the sent email")

        def run(self, _: ToolRunContext, recipient: str, subject: str, body: str) -> str:
            """Mock sending an email and return a confirmation message."""
            # Check first and last line of the chapter are in the body
            nonlocal email_tool_called
            email_tool_called = True
            assert "Well, Prince, so Genoa and Lucca" in body
            assert "I'll start my apprenticeship as old maid" in body
            return f"Email sent to {recipient} with subject '{subject}'"

    plan = (
        PlanBuilder(
            "Send an email to robbie@portialabs.ai titles 'Story' containing the first "
            "chapter of War and Peace"
        )
        .step(
            task="Get the first chapter of War and Peace",
            tool_id=StoryTool().id,
            output="$story",
        )
        .step(
            task="Send an email to robbie@portialabs.ai titles 'Story' containing the first "
            "chapter of War and Peace",
            tool_id=EmailTool().id,
            output="$result",
            inputs=[
                Variable(name="$story", description="The first chapter of War and Peace"),
            ],
        )
        .build()
    )

    portia = Portia(config=config, tools=ToolRegistry([StoryTool(), EmailTool()]))
    plan_run = portia.run_plan(plan)

    assert plan_run.state == PlanRunState.COMPLETE
    assert email_tool_called
