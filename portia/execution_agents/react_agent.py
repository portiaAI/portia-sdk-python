"""A simple OneShotAgent optimized for simple tool calling tasks.

This agent invokes the OneShotToolCallingModel up to four times, but each individual
attempt is a one-shot call. It is useful when the tool call is simple, minimizing cost.
However, for more complex tool calls, the DefaultExecutionAgent is recommended as it will
be more successful than the OneShotAgent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from portia import logger
from portia.execution_agents.execution_utils import AgentNode
from portia.execution_agents.output import LocalDataValue
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.telemetry.views import ExecutionAgentUsageTelemetryEvent

if TYPE_CHECKING:
    from langchain.tools import StructuredTool

    from portia.execution_agents.output import Output
    from portia.model import GenerativeModel
    from portia.run_context import RunContext
    from portia.tool import Tool


class WrappedToolNode(ToolNode):
    """ToolNode subclass that adds logging before and after tool calls."""

    async def ainvoke(self, state: MessagesState) -> dict[str, Any]:
        """Execute tools asynchronously with logging."""
        # @@@ ADD EXECUTION HOOK HANDLING + CLARIFICATION HANDLING

        # Execute the actual tool calls using parent implementation
        result = await super().ainvoke(state)

        # @@@ ADD EXECUTION HOOK HANDLING + CLARIFICATION HANDLING

        return result


class ReActState(MessagesState):
    """State for the execution agent."""

    plan: str


class ReasoningNode:
    """Node that handles planning, reasoning, and tool selection in a unified approach."""

    unified_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                (
                    "You are an autonomous reasoning-and-acting (ReAct) agent. "
                    "{% if is_initial_planning %}"
                    "Your task is to break the given milestone objective into a concise, "
                    "high-level plan "
                    "and then execute it by selecting appropriate tools.\n\n"
                    "First, create a high-level plan consisting of a summary in clear, concise "
                    "language, followed by bullet points for the likely steps needed (each bullet "
                    "on a new line starting with '-'). Then, select the first tool to call to "
                    "begin executing this plan."
                    "{% else %}"
                    "Think step-by-step about the current progress towards the milestone, what is "
                    "still missing, and what actions can be taken to get closer to the goal. "
                    "Then select exactly one tool (and its arguments) that will best progress "
                    "toward completing the milestone."
                    "{% endif %}\n\n"
                    "If you think the goal has already been met or no tool is helpful, provide "
                    "a response without calling any tools."
                ),
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                (
                    "Task:\n"
                    "{% if is_initial_planning %}"
                    "<task>\n{{ task }}\n</task>\n\n"
                    "Additional data available to help with the task:\n"
                    "<task_data>\n{{ task_data }}\n</task_data>\n\n"
                    "{% else %}"
                    "{{ task }}\n\n"
                    "The high level plan for the task is:\n"
                    "<plan>\n{{ initial_plan }}\n</plan>\n\n"
                    "The conversation so far is:\n"
                    "<conversation>\n{{ conversation }}\n</conversation>\n\n"
                    "{% endif %}"
                    "Available tools:\n"
                    "<tools>\n"
                    "{% for tool in tools %}"
                    "<tool>\n"
                    "<tool_id>{{ tool.id }}</tool_id>\n"
                    "<tool_name>{{ tool.name }}</tool_name>\n"
                    "<tool_description>{{ tool.description }}</tool_description>\n"
                    "</tool>\n"
                    "{% endfor %}\n"
                    "</tools>\n\n"
                    "{% if is_initial_planning %}"
                    "As context, the current UTC date is {{ current_date }}. The current UTC "
                    "time is {{ current_time }}.\n\n"
                    "Create the high-level plan and then select the first tool to call."
                    "{% else %}"
                    "Think about the progress and select the best next tool to call."
                    "{% endif %}"
                ),
                template_format="jinja2",
            ),
        ]
    )

    def __init__(
        self,
        task: str,
        task_data: dict[str, Any] | list[Any] | str | None,
        model: GenerativeModel,
        tools: list[Tool],
        langchain_tools: list[StructuredTool],
    ) -> None:
        """Initialize ReasoningNode."""
        self.task = task
        self.task_data = task_data
        self.model = model
        self.tools = tools
        self.langchain_tools = langchain_tools

    async def invoke(self, state: ReActState) -> dict[str, Any]:
        """Handle both initial planning and subsequent reasoning/tool selection."""
        prior_messages: list[BaseMessage] = list(state.get("messages", []))  # type: ignore[arg-type]
        is_initial_planning = len(prior_messages) == 0

        model = self.model.to_langchain().bind_tools(self.langchain_tools)

        # Prepare template variables
        template_vars = {
            "task": self.task,
            "tools": self.tools,
            "is_initial_planning": is_initial_planning,
        }

        if is_initial_planning:
            # First invocation: add planning-specific variables
            template_vars.update(
                {
                    "task_data": self.task_data,
                    "current_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
                    "current_time": datetime.now(tz=UTC).strftime("%H:%M:%S"),
                }
            )
            logger().debug("📋 React Agent Initial Planning Phase")
        else:
            # Subsequent invocations: add reasoning-specific variables
            conversation_text = get_conversation_text(prior_messages, include_last_message=True)
            template_vars.update(
                {
                    "initial_plan": state.get("plan", ""),
                    "conversation": conversation_text,
                }
            )

        # Format and invoke with unified prompt
        formatted_messages = self.unified_prompt.format_messages(**template_vars)
        response = await model.ainvoke(formatted_messages)

        if isinstance(response, AIMessage) and response.tool_calls:
            action = "Initial tool call" if is_initial_planning else "Calling tool"
            logger().info(f"🛠️ {action}: {response.tool_calls[0]['name']}")
        elif isinstance(response, AIMessage) and not response.tool_calls:
            logger().info("👋 No tool calls, exiting")
        else:
            logger().info(f"❌ Unexpected response: {response}")

        logger().info(f"💭 Reasoning: {response.content}")

        # Return appropriate state update
        if is_initial_planning:
            plan_text = response.content if response.content else "Initial plan created"
            return {"messages": [response], "plan": plan_text}

        return {"messages": [response]}


def get_conversation_text(messages: list[BaseMessage], include_last_message: bool = False) -> str:
    """Get the conversation text from the messages."""
    # @@@ UNDERSTAND THIS
    conversation_snippets: list[str] = []
    for i, msg in enumerate(messages):
        should_include = (
            (isinstance(msg, AIMessage) and msg.tool_calls)
            or isinstance(msg, ToolMessage)
            or (include_last_message and i == len(messages) - 1)
        )
        if not should_include:
            continue
        tag = (
            "ToolCall"
            if isinstance(msg, AIMessage) and msg.tool_calls
            else "ToolOutput"
            if isinstance(msg, ToolMessage)
            else "Message"
        )
        conversation_snippets.append(f"<{tag}>{msg.content}</{tag}>")
    return "\n".join(conversation_snippets) if conversation_snippets else "(no prior conversation)"


class ReActAgent:
    """ReAct (Reasoning and Acting) agent that combines planning, reasoning, and tool selection."""

    def __init__(
        self,
        task: str,
        task_data: dict[str, Any] | list[Any] | str | None,
        tools: list[Tool],
        run_data: RunContext,
    ) -> None:
        """Initialize the ReActAgent."""
        self.task = task
        self.task_data = task_data
        self.tools = tools
        self.run_data = run_data

    async def execute(self) -> Output:
        """Run the ReAct agent."""
        self.run_data.telemetry.capture(
            ExecutionAgentUsageTelemetryEvent(
                agent_type="react",
                model=str(self.run_data.config.get_execution_model()),
                sync=False,
                tool_id=",".join([tool.id for tool in self.tools]),
            )
        )

        tool_run_ctx = self.run_data.get_tool_run_ctx()
        model = self.run_data.config.get_execution_model()
        langchain_tools = [
            tool.to_langchain_with_artifact(
                ctx=tool_run_ctx,
            )
            for tool in self.tools
        ]

        graph = StateGraph(ReActState)

        graph.add_node(
            AgentNode.REASONING,
            ReasoningNode(
                task=self.task,
                task_data=self.task_data,
                model=model,
                tools=self.tools,
                langchain_tools=langchain_tools,
            ).invoke,
        )
        graph.add_edge(START, AgentNode.REASONING)
        graph.add_conditional_edges(
            AgentNode.REASONING,
            tool_call_or_summarise,
        )

        graph.add_node(AgentNode.TOOLS, WrappedToolNode(langchain_tools))
        graph.add_edge(AgentNode.TOOLS, AgentNode.REASONING)

        graph.add_node(
            AgentNode.SUMMARIZER,
            StepSummarizer(self.config, model, self.tool, self.step).ainvoke,
        )
        graph.add_edge(AgentNode.SUMMARIZER, END)

        app = graph.compile()
        invocation_result = await app.ainvoke({"messages": []}, config={"recursion_limit": 50})

        return process_output(invocation_result["messages"])


def process_output(messages: list[BaseMessage]) -> Output:
    """Process the output of the agent."""
    # @@@ SORT THIS PROPERLY
    return LocalDataValue(
        summary=str(messages[-1].content) if messages[-1].content else "",
        value=messages[-1].content,
    )


def tool_call_or_summarise(
    state: MessagesState,
) -> Literal[AgentNode.TOOLS, AgentNode.SUMMARIZER]:  # type: ignore  # noqa: PGH003
    """Decide whether to continue with another tool call or transition to summarisation."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # No tool calls means the agent is done and ready to summarize
        return AgentNode.SUMMARIZER
    return AgentNode.TOOLS
