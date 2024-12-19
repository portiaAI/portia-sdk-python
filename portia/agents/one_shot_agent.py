"""One Shot agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from portia.agents.base_agent import BaseAgent
from portia.agents.toolless_agent import ToolLessAgent
from portia.clarification import Clarification
from portia.errors import ToolRetryError
from portia.plan import Output, Variable

if TYPE_CHECKING:
    from langchain.tools import StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.agents.verifier_agent import VerifiedToolInputs
    from portia.tool import Tool


MAX_RETRIES = 4


class OneShotToolCallingModel:
    """Model to call the tool with unverified arguments."""

    tool_calling_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are very powerful assistant, but don't know current events.",
            ),
            HumanMessagePromptTemplate.from_template(
                "query:\n{query}\n"
                "context:\n{context}\n"
                "Make sure you don't repeat past errors: {past_errors}\n"
                "Use the provided tool. You should provide arguments that match the tool's schema\n"
                "using the information contained in the query and context.",
            ),
        ],
    )

    def __init__(
        self,
        llm: BaseChatModel,
        context: str,
        tools: list[StructuredTool],
        agent: OneShotAgent,
    ) -> None:
        """Initialize the model."""
        self.llm = llm
        self.context = context
        self.agent = agent
        self.tools = tools

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        model = self.llm.bind_tools(self.tools)
        messages = state["messages"]
        past_errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        response = model.invoke(
            self.tool_calling_prompt.format_messages(
                query=self.agent.description,
                context=self.context,
                past_errors=past_errors,
            ),
        )

        return {"messages": [response]}


class OneShotAgent(BaseAgent):
    """Agent responsible for achieving a task by using langgraph.

    This agent does the following things:
    1. Calls the tool with unverified arguments.
    2. Retries tool calls up to 4 times.
    """

    def __init__(
        self,
        description: str,
        inputs: list[Variable],
        clarifications: list[Clarification] | None = None,
        tool: Tool | None = None,
        system_context: list[str] | None = None,
    ) -> None:
        """Initialize the agent."""
        super().__init__(description, inputs, clarifications, tool, system_context)
        self.verified_args: VerifiedToolInputs | None = None
        self.single_tool_agent = None
        self.new_clarifications: list[Clarification] = []

    @staticmethod
    def retry_tool_or_finish(state: MessagesState) -> Literal["tool_agent", END]:  # type: ignore  # noqa: PGH003
        """Determine if we should retry calling the tool if there was an error."""
        messages = state["messages"]
        last_message = messages[-1]
        errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        if "ToolSoftError" in last_message.content and len(errors) < MAX_RETRIES:
            return "tool_agent"
        return END

    @staticmethod
    def call_tool_or_return(state: MessagesState) -> Literal["tools", END]:  # type: ignore  # noqa: PGH003
        """Determine if we should continue or not.

        This is only to catch issues when the agent does not figure out how to use the tool
        to achieve the goal.
        """
        last_message = state["messages"][-1]
        # If the LLM makes a tool call, then we route to the "tools" node
        if hasattr(last_message, "tool_calls"):
            return "tools"
        # Otherwise, we stop (reply to the user).
        return END

    def process_output(self, last_message: BaseMessage) -> Output:
        """Process the output of the agent."""
        if "ToolSoftError" in last_message.content and self.tool:
            raise ToolRetryError(self.tool.name, str(last_message.content))
        if len(self.new_clarifications) > 0:
            return Output[list[Clarification]](
                value=self.new_clarifications,
            )
        if isinstance(last_message, ToolMessage):
            if last_message.artifact and isinstance(last_message.artifact, Output):
                tool_output = last_message.artifact
            elif last_message.artifact:
                tool_output = Output(value=last_message.artifact)
            else:
                tool_output = Output(value=last_message.content)
            return tool_output
        if isinstance(last_message, HumanMessage):
            return Output(value=last_message.content)
        return Output(value="Requested clarification or failed to call tool.")

    def execute_sync(self, llm: BaseChatModel, step_outputs: dict[str, Output]) -> Output:
        """Run the core execution logic of the task."""
        if not self.tool:
            if not self.single_tool_agent:
                self.single_tool_agent = ToolLessAgent(
                    self.description,
                    self.inputs,
                    self.clarifications,
                    system_context=self.system_context,
                )
            return self.single_tool_agent.execute_sync(llm, step_outputs)

        context = self._get_context(step_outputs)

        tools = [self.tool.to_langchain(return_artifact=True)]
        tool_node = ToolNode(tools)

        workflow = StateGraph(MessagesState)
        workflow.add_node("tool_agent", OneShotToolCallingModel(llm, context, tools, self).invoke)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "tool_agent")
        workflow.add_conditional_edges("tool_agent", self.call_tool_or_return)

        workflow.add_conditional_edges(
            "tools",
            OneShotAgent.retry_tool_or_finish,
        )

        # We could use a MemorySaver checkpointer to hold intermediate state,
        # this will allow us to process clarifications as if they were regular
        # returns from tool calls.
        app = workflow.compile()

        invocation_result = app.invoke({"messages": []})
        return self.process_output(invocation_result["messages"][-1])
