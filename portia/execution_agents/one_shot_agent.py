"""A simple OneShotAgent optimized for simple tool calling tasks.

This agent invokes the OneShotToolCallingModel up to four times, but each individual
attempt is a one-shot call. It is useful when the tool call is simple, minimizing cost.
However, for more complex tool calls, the DefaultExecutionAgent is recommended as it will
be more successful than the OneShotAgent.
"""

from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph.prebuilt import ToolNode
from portia.errors import InvalidAgentError
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.execution_utils import (
    AgentNode,
    process_output,
    tool_call_or_end,
)
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.execution_context import get_execution_context
from portia.tool import ToolRunContext
from portia.execution_agents.context import StepInput  # noqa: TC001
from portia.plan import Step, ReadOnlyStep
from portia.plan_run import PlanRun, ReadOnlyPlanRun


if TYPE_CHECKING:
    from langchain.tools import StructuredTool

    from portia.config import Config
    from portia.end_user import EndUser
    from portia.execution_agents.output import Output
    from portia.execution_hooks import ExecutionHooks
    from portia.model import GenerativeModel
    from portia.plan import Step
    from portia.plan_run import PlanRun
    from portia.storage import AgentMemory
    from portia.tool import Tool


class ExecutionState(MessagesState):
    """State for the execution agent."""

    step_inputs: list[StepInput]


class OneShotToolCallingModel:
    """One-shot model for calling a given tool.

    This model directly passes the tool and context to the language model (LLM)
    to generate a response. It is suitable for simple tasks where the arguments
    are already correctly formatted and complete. This model does not validate
    arguments (e.g., it will not catch missing arguments).

    It is recommended to use the DefaultExecutionAgent for more complex tasks.

    Args:
        model (GenerativeModel): The language model to use for generating responses.
        tools (list[StructuredTool]): A list of tools that can be used during the task.
        agent (OneShotAgent): The agent responsible for managing the task.

    Methods:
        invoke(MessagesState): Invokes the LLM to generate a response based on the query, context,
                               and past errors.

    """

    tool_calling_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a very powerful assistant, but don't know current events.",
            ),
            HumanMessagePromptTemplate.from_template(
                [
                    "query:",
                    "{query}",
                    "context:",
                    "{context}",
                    "Use the provided tool. You should provide arguments that match the tool's "
                    "schema using the information contained in the query and context."
                    "Important! Make sure to take into account previous clarifications in the "
                    "context which are from the user and may change the query"
                    "Make sure you don't repeat past errors: {past_errors}",
                ],
            ),
        ],
    )

    def __init__(
        self,
        model: GenerativeModel,
        tools: list[StructuredTool],
        agent: OneShotAgent,
        tool_context: ToolRunContext,
    ) -> None:
        """Initialize the OneShotToolCallingModel.

        Args:
            model (GenerativeModel): The language model to use for generating responses.
            tools (list[StructuredTool]): A list of tools that can be used during the task.
            agent (OneShotAgent): The agent that is managing the task.
            tool_context (ToolRunContext): The context for the tool.

        """
        self.model = model
        self.agent = agent
        self.tools = tools
        self.tool_context = tool_context

    def invoke(self, state: ExecutionState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        This method formats the input for the language model using the query, context,
        and past errors, then generates a response by invoking the model.

        Args:
            state (ExecutionState): The state containing the messages and other necessary data.

        Returns:
            dict[str, Any]: A dictionary containing the model's generated response.

        """
        model = self.model.to_langchain().bind_tools(self.tools)
        messages = state["messages"]
        past_errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        context = self.agent.get_system_context(self.tool_context, state["step_inputs"])
        response = model.invoke(
            self.tool_calling_prompt.format_messages(
                query=self.agent.step.task,
                context=context,
                past_errors=past_errors,
            ),
        )
        if (
            self.agent.execution_hooks
            and self.agent.execution_hooks.before_tool_call
            and self.agent.tool
        ):
            for tool_call in response.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
                clarification = self.agent.execution_hooks.before_tool_call(
                    self.agent.tool,
                    tool_call.get("args"),
                    ReadOnlyPlanRun.from_plan_run(self.agent.plan_run),
                    ReadOnlyStep.from_step(self.agent.step),
                )
                if clarification:
                    self.agent.new_clarifications.append(clarification)
                    return {"messages": []}

        return {"messages": [response]}


class OneShotAgent(BaseExecutionAgent):
    """Agent responsible for achieving a task by using langgraph.

    This agent performs the following steps:
    1. Extracts inputs from agent memory (if applicable)
    2. Calls the tool with unverified arguments.
    3. Retries tool calls up to 4 times.

    Methods:
        execute_sync(): Executes the core logic of the agent's task, using the provided tool

    """

    def __init__(  # noqa: PLR0913
        self,
        step: Step,
        plan_run: PlanRun,
        config: Config,
        agent_memory: AgentMemory,
        end_user: EndUser,
        tool: Tool | None = None,
        execution_hooks: ExecutionHooks | None = None,
    ) -> None:
        """Initialize the OneShotAgent.

        Args:
            step (Step): The current step in the task plan.
            plan_run (PlanRun): The run that defines the task execution process.
            config (Config): The configuration settings for the agent.
            agent_memory (AgentMemory): The agent memory for persisting outputs.
            end_user (EndUser): The end user for the execution.
            tool (Tool | None): The tool to be used for the task (optional).
            execution_hooks (ExecutionHooks | None): The execution hooks for the agent.

        """
        super().__init__(step, plan_run, config, end_user, agent_memory, tool, execution_hooks)

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task.

        This method will invoke the tool with arguments

        Returns:
            Output: The result of the agent's execution, containing the tool call result.

        """
        if not self.tool:
            raise InvalidAgentError("No tool available")

        tool_run_ctx = ToolRunContext(
            execution_context=get_execution_context(),
            end_user=self.end_user,
            plan_run_id=self.plan_run.id,
            config=self.config,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )

        model = self.config.get_execution_model()
        tools = [
            self.tool.to_langchain_with_artifact(
                ctx=tool_run_ctx,
            ),
        ]
        tool_node = ToolNode(tools)

        graph = StateGraph(ExecutionState)
        graph.add_node(AgentNode.MEMORY_EXTRACTION, MemoryExtractionStep(self).invoke)
        graph.add_edge(START, AgentNode.MEMORY_EXTRACTION)

        graph.add_node(
            AgentNode.TOOL_AGENT,
            OneShotToolCallingModel(model, tools, self, tool_run_ctx).invoke,
        )
        graph.add_edge(AgentNode.MEMORY_EXTRACTION, AgentNode.TOOL_AGENT)

        graph.add_node(AgentNode.TOOLS, tool_node)
        graph.add_node(
            AgentNode.SUMMARIZER,
            StepSummarizer(self.config, model, self.tool, self.step).invoke,
        )

        # Use execution manager for state transitions
        graph.add_conditional_edges(
            AgentNode.TOOL_AGENT,
            tool_call_or_end,
        )
        graph.add_conditional_edges(
            AgentNode.TOOLS,
            lambda state: self.next_state_after_tool_call(self.config, state, self.tool),
        )
        graph.add_edge(AgentNode.SUMMARIZER, END)

        app = graph.compile()
        invocation_result = app.invoke({"messages": [], "step_inputs": []})

        return process_output(invocation_result["messages"], self.tool, self.new_clarifications)
