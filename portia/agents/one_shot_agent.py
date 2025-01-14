"""A simple OneShotAgent that is optimized for simple tool calling tasks.

It invokes the OneShotToolCallingModel up to four times but each individual attempt is a one shot.
This agent is useful when the tool call is simple as it minimizes cost, but the VerifierAgent will
be more successful on anything but simple tool calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from portia.agents.base_agent import BaseAgent, Output
from portia.agents.toolless_agent import ToolLessAgent
from portia.clarification import Clarification
from portia.context import get_execution_context
from portia.errors import InvalidAgentOutputError, ToolFailedError, ToolRetryError
from portia.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from langchain.tools import StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.agents.verifier_agent import VerifiedToolInputs
    from portia.config import Config
    from portia.plan import Step
    from portia.tool import Tool
    from portia.workflow import Workflow


# MAX_RETRIES controls how many times errors will be retried by the OneShotAgent
MAX_RETRIES = 4


class OneShotToolCallingModel:
    """OneShotToolCallingModel is a one shot model for calling the given tool.

    The tool and context are given directly to the LLM and we return the results.
    This model is useful for simple tasks where the arguments are in the correct form
    and are all present. The OneShotToolModel will not carry out validation of arguments,
    for example it will not complain about missing arguments.

    Prefer to use the VerifierAgent if you have more complicated needs.
    """

    tool_calling_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are very powerful assistant, but don't know current events.",
            ),
            HumanMessagePromptTemplate.from_template(
                [
                    "query:",
                    "{query}",
                    "context:",
                    "{context}",
                    "Use the provided tool. You should provide arguments that match the tool's"
                    "schema using the information contained in the query and context."
                    "Where clarifications have been provided in the context you should always"
                    "use the values provided by them.",
                    "Make sure you don't repeat past errors: {past_errors}",
                ],
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
                query=self.agent.step.task,
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
        step: Step,
        workflow: Workflow,
        config: Config,
        tool: Tool | None = None,
    ) -> None:
        """Initialize the agent."""
        super().__init__(step, workflow, config, tool)
        self.verified_args: VerifiedToolInputs | None = None
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
        if "ToolHardError" in last_message.content and self.tool:
            raise ToolFailedError(self.tool.name, str(last_message.content))
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
        raise InvalidAgentOutputError(str(last_message.content))

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task."""
        if not self.tool:
            single_tool_agent = ToolLessAgent(
                self.step,
                self.workflow,
                self.config,
                self.tool,
            )
            return single_tool_agent.execute_sync()

        context = self.get_system_context()
        llm = LLMWrapper(self.config).to_langchain()

        tools = [
            self.tool.to_langchain(
                return_artifact=True,
                ctx=get_execution_context(),
            ),
        ]
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

        app = workflow.compile()

        invocation_result = app.invoke({"messages": []})

        return self.process_output(invocation_result["messages"][-1])
