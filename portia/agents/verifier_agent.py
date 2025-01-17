"""The Verifier Agent for hardest problems.

This agent uses multiple models (verifier, parser etc) to achieve the highest accuracy
in completing tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portia.agents.base_agent import BaseAgent, Output
from portia.agents.toolless_agent import ToolLessAgent
from portia.clarification import Clarification, InputClarification
from portia.context import get_execution_context
from portia.errors import (
    InvalidAgentOutputError,
    InvalidWorkflowStateError,
    ToolFailedError,
    ToolRetryError,
)
from portia.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from langchain.tools import StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.config import Config
    from portia.plan import Step
    from portia.tool import Tool
    from portia.workflow import Workflow

MAX_RETRIES = 4


class ToolArgument(BaseModel):
    """Represents an argument for a tool as extracted from the goal and context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the argument, as requested by the tool.")
    value: Any | None = Field(
        description="Value of the argument, as provided by in the goal or context.",
    )
    valid: bool = Field(
        description="Whether the value is a valid type and or format for the given argument.",
    )
    explanation: str = Field(description="Explanation of the source for the value of the argument.")


class ToolInputs(BaseModel):
    """Represents the inputs for a tool."""

    args: list[ToolArgument] = Field(description="Arguments for the tool.")


class VerifiedToolArgument(BaseModel):
    """Represents an argument for a tool after being verified by an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the argument, as requested by the tool.")
    value: Any | None = Field(
        description="Value of the argument, as provided by in the goal or context.",
    )

    # We call this "made_up" and not "hallucinated" because the latter was making OpenAI's model
    # produce invalid JSON.
    made_up: bool = Field(
        description="Whether the value was made up or not. "
        "Should be false if the value was provided by the user, even if in a different format."
        "User provided values can be in the context, in the goal or the result of previous steps.",
    )


class VerifiedToolInputs(BaseModel):
    """Represents the inputs for a tool."""

    args: list[VerifiedToolArgument] = Field(description="Arguments for the tool.")


class ParserModel:
    """Model to parse the arguments for a tool."""

    arg_parser_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a highly capable assistant tasked with generating valid arguments for "
                    "tools based on provided input. "
                    "While you are not aware of current events, you excel at reasoning "
                    "and adhering to instructions. "
                    "Your responses must clearly explain the source of each argument "
                    "(e.g., context, past messages, clarifications). "
                    "Avoid assumptions or fabricated information."
                ),
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:\n{context}\n"
                "Task: {task}\n"
                "The system has a tool available named '{tool_name}'.\n"
                "Argument schema for the tool:\n{tool_args}\n"
                "Description of the tool: {tool_description}\n"
                "\n\n----------\n\n"
                "The following section contains previous schema errors. "
                "Ensure your response avoids these errors:\n"
                "{previous_errors}\n"
                "\n\n----------\n\n"
                "Please provide the arguments for the tool. Adhere to the following guidelines:\n"
                "- You may take values from the task, inputs, previous steps or clarifications\n"
                "- Prefer values clarified in follow-up inputs over initial inputs.\n"
                "- Do not provide placeholder values (e.g., 'example@example.com').\n"
                "- Ensure arguments align with the tool's schema and intended use.\n",
            ),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: VerifierAgent) -> None:
        """Initialize the model."""
        self.llm = llm
        self.context = context
        self.agent = agent
        self.previous_errors: list[str] = []
        self.retries = 0

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        if not self.agent.tool:
            raise InvalidWorkflowStateError(None)
        model = self.llm.with_structured_output(ToolInputs)
        response = model.invoke(
            self.arg_parser_prompt.format_messages(
                context=self.context,
                task=self.agent.step.task,
                tool_name=self.agent.tool.name,
                tool_args=self.agent.tool.args_json_schema(),
                tool_description=self.agent.tool.description,
                previous_errors=",".join(self.previous_errors),
            ),
        )
        response = ToolInputs.model_validate(response)

        # also test the ToolInputs that have come back
        # actually work for the schema of the tool
        # if not we can retry
        test_args = {}
        for arg in response.args:
            test_args[arg.name] = arg.value
        try:
            self.agent.tool.args_schema.model_validate(test_args)
        except ValidationError as e:
            err = str(e)
            self.previous_errors.append(err)
            self.retries += 1
            if self.retries <= MAX_RETRIES:
                return self.invoke(state)
            raise InvalidAgentOutputError(err) from e

        return {"messages": [response.model_dump_json(indent=2)]}


class VerifierModel:
    """Model to verify the arguments for a tool."""

    arg_verifier_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an expert reviewer. Your task is to validate and label arguments "
                "provided. You must return the made_up field based "
                "on the rules below.\n - An argument is made up if we cannot tell where the value "
                "came from in the provided context\n- Do not just trust the explanations provided"
                "\n- If an argument is marked as invalid it is likely wrong."
                "\n- We really care if the value of an argument is not in the context, a handled "
                "clarification or goal at all (then made_up should be TRUE), but it is ok if "
                "it is there but in a different format (then made_up should be FALSE). "
                "\n- Arguments where the value comes from a clarification should be marked as FALSE"
                "\nThe output must conform to the following schema:\n\n"
                "class VerifiedToolArgument:\n"
                "  name: str  # Name of the argument requested by the tool.\n"
                "  value: Any | None  # Value of the argument from the goal or context.\n"
                "  made_up: bool  # if the value is made_up based on the given rules.\n\n"
                "class VerifiedToolInputs:\n"
                "  args: List[VerifiedToolArgument]  # List of tool arguments.\n\n"
                "Please ensure the output matches the VerifiedToolInputs schema.",
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:"
                "\n{context}\n"
                "You will need to achieve the following goal: {task}\n"
                "\n\n----------\n\n"
                "Label of the following arguments as made up or not: {arguments}\n",
            ),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: VerifierAgent) -> None:
        """Initialize the model."""
        self.llm = llm
        self.context = context
        self.agent = agent

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        messages = state["messages"]
        tool_args = messages[-1].content

        model = self.llm.with_structured_output(VerifiedToolInputs)
        response = model.invoke(
            self.arg_verifier_prompt.format_messages(
                context=self.context,
                task=self.agent.step.task,
                arguments=tool_args,
            ),
        )
        response = VerifiedToolInputs.model_validate(response)
        self.agent.verified_args = response
        return {"messages": [response.model_dump_json(indent=2)]}


class ToolCallingModel:
    """Model to call the tool with the verified arguments."""

    tool_calling_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are very powerful assistant, but don't know current events.",
            ),
            HumanMessagePromptTemplate.from_template(
                "context:\n{verified_args}\n"
                "Make sure you don't repeat past errors: {past_errors}\n"
                "Use the provided tool with the arguments in the context, as "
                "long as they are valid.\n",
            ),
        ],
    )

    def __init__(
        self,
        llm: BaseChatModel,
        context: str,
        tools: list[StructuredTool],
        agent: VerifierAgent,
    ) -> None:
        """Initialize the model."""
        self.llm = llm
        self.context = context
        self.agent = agent
        self.tools = tools

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        verified_args = self.agent.verified_args
        if not verified_args:
            raise InvalidWorkflowStateError
        # handle any clarifications before calling
        if self.agent and self.agent.workflow.outputs.clarifications:
            for arg in verified_args.args:
                matching_clarification = self.agent.get_last_resolved_clarification(arg.name)
                if matching_clarification and arg.value != matching_clarification.response:
                    arg.value = matching_clarification.response
                    arg.made_up = False

        model = self.llm.bind_tools(self.tools)

        messages = state["messages"]
        past_errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        response = model.invoke(
            self.tool_calling_prompt.format_messages(
                verified_args=verified_args.model_dump_json(indent=2),
                past_errors=past_errors,
            ),
        )
        return {"messages": [response]}


class VerifierAgent(BaseAgent):
    """Agent responsible for achieving a task by using verification.

    This agent does the following things:
     1. It uses an LLM to make sure that we have the right arguments for the tool, with
        explanations of the values and where they come from.
     2. It uses an LLM to make sure that the arguments are correct, and that they are labeled
        as provided, inferred or assumed.
     3. If any of the arguments are assumed, it will request a clarification.
     4. If the arguments are correct, it will call the tool and return the result to the user.
     5. If the tool fails, it will try again at least 3 times.

    Also, if the agent is being called a second time, it will just jump to step 4.

    Possible improvements:
     1. This approach (as well as the other agents) could be improved for arguments that are lists
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
        # Otherwise, we stop (reply to the user) as its a hard error or unknown
        return END

    def clarifications_or_continue(self, state: MessagesState) -> Literal["tool_agent", END]:  # type: ignore  # noqa: PGH003
        """Determine if we should continue with the tool call or request clarifications instead."""
        messages = state["messages"]
        last_message = messages[-1]
        arguments = VerifiedToolInputs.model_validate_json(str(last_message.content))

        for arg in arguments.args:
            if not arg.made_up:
                continue
            matching_clarification = self.get_last_resolved_clarification(arg.name)

            if not matching_clarification:
                self.new_clarifications.append(
                    InputClarification(
                        argument_name=arg.name,
                        user_guidance=f"Missing Argument: {arg.name}",
                    ),
                )
        if len(self.new_clarifications) > 0:
            return END

        state.update({"messages": [arguments.model_dump_json(indent=2)]})  # type: ignore  # noqa: PGH003
        return "tool_agent"

    def get_last_resolved_clarification(
        self,
        arg_name: str,
    ) -> Clarification | None:
        """Get the last resolved clarification for an argument."""
        matching_clarification = None
        for clarification in self.workflow.outputs.clarifications:
            if (
                clarification.resolved
                and getattr(clarification, "argument_name", None) == arg_name
                and clarification.step == self.workflow.current_step_index
            ):
                matching_clarification = clarification
        return matching_clarification

    @staticmethod
    def call_tool_or_return(state: MessagesState) -> Literal["tools", END]:  # type: ignore  # noqa: PGH003
        """Determine if we should continue or not.

        This is only to catch issues when the agent does not figure out how to use the tool
        to achieve the goal.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls"):
            return "tools"
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
        """
        The execution graph represented here can be generated using
        `print(app.get_graph().draw_mermaid())` on the compiled workflow (and running any agent
        task). The below represents the current state of the graph (use a mermaid editor
        to view e.g <https://mermaid.live/edit>)

        graph TD;
            __start__([<p>__start__</p>]):::first
            tool_agent(tool_agent)
            argument_parser(argument_parser)
            argument_verifier(argument_verifier)
            tools(tools)
            __end__([<p>__end__</p>]):::last
            __start__ --> argument_parser;
            argument_parser --> argument_verifier;
            tool_agent --> tools;
            argument_verifier -.-> tool_agent;
            argument_verifier -.-> __end__;
            tools -.-> tool_agent;
            tools -.-> __end__;
            classDef default fill:#f2f0ff,line-height:1.2
            classDef first fill-opacity:0
            classDef last fill:#bfb6fc
        """

        workflow.add_node("tool_agent", ToolCallingModel(llm, context, tools, self).invoke)
        if self.verified_args:
            workflow.add_edge(START, "tool_agent")
        else:
            workflow.add_node("argument_parser", ParserModel(llm, context, self).invoke)
            workflow.add_node("argument_verifier", VerifierModel(llm, context, self).invoke)
            workflow.add_edge(START, "argument_parser")
            workflow.add_edge("argument_parser", "argument_verifier")

            workflow.add_conditional_edges(
                "argument_verifier",
                self.clarifications_or_continue,
            )

        workflow.add_node("tools", tool_node)

        workflow.add_conditional_edges("tool_agent", self.call_tool_or_return)

        workflow.add_conditional_edges(
            "tools",
            VerifierAgent.retry_tool_or_finish,
        )

        app = workflow.compile()

        invocation_result = app.invoke({"messages": []})

        return self.process_output(invocation_result["messages"][-1])
