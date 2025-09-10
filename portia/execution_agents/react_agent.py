"""A simple ReAct agent that reasons and acts (using tools) in a loop until the task is complete."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, create_model

from portia.errors import InvalidAgentError, ToolHardError, ToolNotFoundError, ToolSoftError
from portia.execution_agents.execution_utils import AgentNode, is_clarification
from portia.execution_agents.output import LocalDataValue
from portia.execution_agents.react_clarification_tool import ReActClarificationTool
from portia.logger import logger, truncate_message
from portia.model import Message
from portia.telemetry.views import ExecutionAgentUsageTelemetryEvent
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.tools import StructuredTool

    from portia.clarification import Clarification
    from portia.execution_agents.output import Output
    from portia.model import GenerativeModel
    from portia.run_context import RunContext


class WrappedToolNode(ToolNode):
    """ToolNode subclass that adds logging before and after tool calls."""

    def __init__(
        self, run_data: RunContext, tools: list[Tool], langchain_tools: list[StructuredTool]
    ) -> None:
        """Initialize WrappedToolNode."""
        super().__init__(langchain_tools)
        self.run_data = run_data
        self.tools = tools

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:  # noqa: A002, ANN401
        """Execute tools asynchronously with logging."""
        last_message = input["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            raise ToolSoftError(  # pragma: no cover - we should never end up here
                "ReAct agent tried to use a tool without specifying the call to make"
            )

        tool_names = set()
        tool = None
        tool_name = None
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "unknown tool")
            tool_names.add(tool_name)
            tool = self.get_tool_by_name(tool_name)
        if not tool:
            raise ToolNotFoundError(
                f"Tool {tool_name} not found"
            )  # pragma: no cover - we should never end up here

        step = self.run_data.plan.steps[self.run_data.plan_run.current_step_index]

        if self.run_data.execution_hooks.before_tool_call:
            logger().debug("Calling before_tool_call execution hook")
            clarification = self.run_data.execution_hooks.before_tool_call(
                tool, kwargs, self.run_data.plan_run, step.to_legacy_step(self.run_data.plan)
            )
            if clarification:
                return {
                    "messages": ToolMessage(
                        content=str(clarification),
                        artifact=LocalDataValue(value=clarification),
                        tool_call_id=last_message.tool_calls[0]["id"],
                        name=tool.get_langchain_name(),
                        status="success",
                    )
                }

        logger().info(f"🛠️ Calling tool{'s' if len(tool_names) > 1 else ''}: {','.join(tool_names)}")
        result = await super().ainvoke(input, config, **kwargs)
        logger().debug(f"🛠️ Tool result: {result}")

        if self.run_data.execution_hooks.after_tool_call:
            logger().debug("Calling after_tool_call execution hook")
            self.run_data.execution_hooks.after_tool_call(
                tool, result, self.run_data.plan_run, step.to_legacy_step(self.run_data.plan)
            )

        return result

    def get_tool_by_name(self, name: str) -> Tool | None:
        """Get the tool by the name."""
        return next((tool for tool in self.tools if tool.get_langchain_name() == name), None)


class FinalResultToolSchema(BaseModel):
    """Schema defining the inputs for the FinalResultTool."""

    final_result: str = Field(description="The complete final result or result for the task.")


class FinalResultTool(Tool[str]):
    """Tool for providing the final result when a task is completed."""

    id: str = "final_result_tool"
    name: str = "Final Result Tool"
    description: str = (
        "Call this tool when you have completed the task and want to provide the final result."
    )
    args_schema: type[BaseModel] = FinalResultToolSchema
    output_schema: tuple[str, str] = (
        "str",
        "The final result of the task",
    )

    def run(self, ctx: ToolRunContext, final_result: str) -> str:  # noqa: ARG002
        """Run the FinalResultTool."""
        logger().info(f"Final result returned by ReAct agent: {truncate_message(final_result)}")
        return final_result


class ReasoningNode:
    """Node that handles planning, reasoning, and tool selection in a unified approach."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                (
                    "You are an excellent, helpful, autonomous reasoning-and-acting (ReAct) agent "
                    "that completes tasks using reasoning and tools. In order to complete a task, "
                    "you use tools and reason step-by-step about the task in the following loop:\n"
                    "1. Reason about the task, including analyzing any previous tool call "
                    "results and previous conversation, to determine what actions are needed to "
                    "complete the task (Reasoning).\n"
                    "2. If required, select one tool (and its arguments) to call in order to make "
                    "progress toward completing the task (Acting). You should select the tool that "
                    "will best progress towards completing the task. This tool will then be "
                    "called and the result of the tool call will be returned to you.\n"
                    "3. Repeat this in a loop until you are ready to provide the final result. "
                    "When you are ready to provide the final result, use the final_result_tool "
                    "to return the result. This result can either be a final answer (if the task "
                    "is a query with an answer), or it can be confirmation that the required "
                    "actions were taken (if the task instead comprises a series of actions that "
                    "must be taken to complete the task)\n\n"
                    "For the task, you will receive the task description along with a list of "
                    "tools you can use and any previous conversation that has happened in "
                    "completing this task (e.g. previous reasoning, tool call results etc.). "
                    "You must determine what the correct next step is based on this information. "
                    "You should format your output as follows:\n"
                    "* If reasoning, write your thoughts in natural language\n"
                    "* If taking an action (i.e. calling a tool), return a JSON tool call\n"
                    "* If ready to give the final answer, return a JSON tool call that calls the "
                    "final_result_tool with the final result as the argument.\n"
                ),
            ),
            HumanMessagePromptTemplate.from_template(
                (
                    "Task Description:\n"
                    "<task>\n{{ task }}\n</task>\n\n"
                    "Additional data available to help with the task:\n"
                    "<task_data>\n{{ task_data }}\n</task_data>\n\n"
                    "The conversation so far is below (most recent messages at the end):\n"
                    "<conversation>\n{{ conversation }}\n</conversation>\n\n"
                    "{% if clarifications %}"
                    "The result of previous clarifications with the user are:\n"
                    "{% for clarification in clarifications %}"
                    "<clarification>\n{{ clarification }}\n</clarification>\n\n"
                    "{% endfor %}\n"
                    "{% endif %}"
                    "Available tools:\n"
                    "<tools>\n"
                    "{% for tool in tools %}"
                    "<tool>\n"
                    "<tool_id>{{ tool.id }}</tool_id>\n"
                    "<tool_name>{{ tool.name }}</tool_name>\n"
                    "<tool_description>{{ tool.description }}</tool_description>\n"
                    "<tool_args_schema>{{ tool.args_schema }}</tool_args_schema>\n"
                    "<tool_output_schema>{{ tool.output_schema }}</tool_output_schema>\n"
                    "</tool>\n"
                    "{% endfor %}"
                    "</tools>\n\n"
                    "As context, the current UTC date is {{ current_date }}. The current UTC "
                    "time is {{ current_time }}.\n\n"
                    "Please use the information above to determine the best next step in "
                    "completing the task, or call the final_result_tool with the final result if "
                    "you are ready to provide the final result."
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
        prev_clarifications: list[Clarification],
    ) -> None:
        """Initialize ReasoningNode."""
        self.task = task
        self.task_data = task_data
        self.model = model
        self.tools = tools
        self.langchain_tools = langchain_tools
        self.prev_clarifications = prev_clarifications

    async def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Run the reasoning step of the ReAct agent."""
        prior_messages: list[BaseMessage] = list(state.get("messages", []))  # type: ignore[arg-type]

        model = self.model.to_langchain().bind_tools(self.langchain_tools)
        clarifications = [
            self._get_clarification_text(clarification)
            for clarification in self.prev_clarifications
        ]
        template_vars = {
            "task": self.task,
            "task_data": self.task_data,
            "conversation": self._get_conversation_text(prior_messages),
            "tools": self.tools,
            "current_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            "current_time": datetime.now(tz=UTC).strftime("%H:%M:%S"),
            "clarifications": clarifications,
        }

        last_message = prior_messages[-1].content if prior_messages else self.task
        logger().info(f"💭 Reasoning - last message: {truncate_message(last_message)}")
        formatted_messages = self.prompt.format_messages(**template_vars)
        response = await model.ainvoke(formatted_messages)
        logger().debug(f"💭 Reasoning - response: {truncate_message(response)}")

        return {"messages": [response]}

    def _get_conversation_text(self, messages: list[BaseMessage]) -> str:
        """Get the conversation text from the messages."""
        conversation_snippets: list[str] = []
        for msg in messages:
            tag = (
                "ToolCall"
                if isinstance(msg, AIMessage) and msg.tool_calls
                else "ToolOutput"
                if isinstance(msg, ToolMessage)
                else "Message"
            )
            msg_content = (
                msg.tool_calls if isinstance(msg, AIMessage) and msg.tool_calls else msg.content
            )
            conversation_snippets.append(f"<{tag}>{msg_content}</{tag}>")
        return (
            "\n".join(conversation_snippets) if conversation_snippets else "(no prior conversation)"
        )

    def _get_clarification_text(self, clarification: Clarification) -> str:
        """Get the clarifications text from the clarifications."""
        return (
            f"Clarification with guidance '{clarification.user_guidance}' "
            f"had response from user: {clarification.response}"
        )


class ReActAgent:
    """ReAct (Reasoning and Acting) agent that combines planning, reasoning, and tool selection."""

    def __init__(
        self,
        task: str,
        task_data: dict[str, Any] | list[Any] | str | None,
        tools: Sequence[Tool],
        run_data: RunContext,
        tool_call_limit: int = 25,
        allow_agent_clarifications: bool = False,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize the ReActAgent."""
        self.task = task
        self.task_data = task_data
        self.tools = list(tools)
        self.run_data = run_data
        self.tool_call_limit = tool_call_limit
        self.allow_agent_clarifications = allow_agent_clarifications
        self.output_schema = output_schema

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
        # We use the planning model rather than the execution model here as
        # ReAct agents have to both plan and execute, but the planning model
        # is generally more powerful.
        model = self.run_data.config.get_planning_model()
        self.tools.append(FinalResultTool())
        if self.allow_agent_clarifications:
            self.tools.append(ReActClarificationTool())
        langchain_tools = [
            tool.to_langchain_with_artifact(
                ctx=tool_run_ctx,
            )
            for tool in self.tools
        ]

        graph = StateGraph(MessagesState)

        graph.add_node(
            AgentNode.REASONING,
            ReasoningNode(
                task=self.task,
                task_data=self.task_data,
                model=model,
                tools=self.tools,
                langchain_tools=langchain_tools,
                prev_clarifications=self.run_data.plan_run.get_clarifications_for_step(),
            ).invoke,
        )
        graph.add_edge(START, AgentNode.REASONING)
        graph.add_edge(AgentNode.REASONING, AgentNode.TOOLS)

        graph.add_node(AgentNode.TOOLS, WrappedToolNode(self.run_data, self.tools, langchain_tools))
        graph.add_conditional_edges(
            AgentNode.TOOLS,
            next_state_after_tool_call,
        )

        app = graph.compile()
        try:
            invocation_result = await app.ainvoke(
                # Multiply by 2 as each tool call has 2 steps (reasoning and tool call)
                {"messages": []},
                config={"recursion_limit": self.tool_call_limit * 2},
            )
        except GraphRecursionError as e:
            raise InvalidAgentError("ReAct agent reached tool call limit") from e

        return self.process_output(invocation_result["messages"])

    def process_output(self, messages: list[BaseMessage]) -> Output:
        """Process the output of the agent."""
        for message in reversed(messages):
            if (
                isinstance(message, ToolMessage)
                and message.name == FinalResultTool().get_langchain_name()
                and message.artifact
                and isinstance(message.artifact, LocalDataValue)
            ):
                return self._create_output_with_summary(message.artifact)

        if messages and (last_message := messages[-1]):
            if (
                isinstance(last_message, ToolMessage)
                and isinstance(last_message.artifact, LocalDataValue)
                and is_clarification(last_message.artifact.value)
            ):
                return last_message.artifact
            # Fallback to the last message content
            return LocalDataValue(  # pragma: no cover - we should never end up here
                value=last_message.content,
                summary=f"Task {self.task} finished with result: {last_message.content!s}",
            )
        raise ToolHardError(
            "React agent completed without result"
        )  # pragma: no cover - we should never end up here

    def _create_output_with_summary(self, final_result: LocalDataValue) -> LocalDataValue:
        """Create output with summary and handle structured output if needed."""
        if self.output_schema:
            summarizer_output_model = create_model(
                "SummarizerOutput",
                __base__=self.output_schema,
                so_summary=(str, Field(description="A summary of the final result")),
            )

            result = self.run_data.config.get_summarizer_model().get_structured_response(
                [
                    Message(
                        content=(
                            f"The task '{self.task}' had final result: {final_result.value}. "
                            "Please coerce the final result to the output schema and add a summary "
                            "of the final result."
                        ),
                        role="user",
                    )
                ],
                summarizer_output_model,
            )
            processed_result = self.output_schema.model_validate(result.model_dump())
            summary = result.so_summary  # pyright: ignore[reportAttributeAccessIssue]
        else:
            processed_result = final_result.value
            summary_result = self.run_data.config.get_summarizer_model().get_response(
                [
                    Message(
                        content=(
                            f"Summarize the following final result to the task '{self.task}': "
                            f"{final_result.value}"
                        ),
                        role="user",
                    )
                ]
            )
            summary = summary_result.content

        return LocalDataValue(
            value=processed_result,
            summary=summary,
        )


def next_state_after_tool_call(
    state: MessagesState,
) -> Literal[END, AgentNode.REASONING]:  # type: ignore  # noqa: PGH003
    """Decide the next state after a tool call."""
    last_message = state["messages"][-1]
    if (
        last_message
        and isinstance(last_message, ToolMessage)
        and isinstance(last_message.artifact, LocalDataValue)
        and is_clarification(last_message.artifact.value)
    ):
        return END

    if (
        isinstance(last_message, ToolMessage)
        and last_message.name == FinalResultTool().get_langchain_name()
    ):
        return END

    return AgentNode.REASONING
