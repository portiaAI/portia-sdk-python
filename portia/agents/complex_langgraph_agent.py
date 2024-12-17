"""Complex Agent for hardest problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portia.agents.base_agent import BaseAgent
from portia.agents.toolless_agent import ToolLessAgent
from portia.clarification import ArgumentClarification, Clarification, InputClarification
from portia.plan import Output, Variable
from portia.tool import ToolFailedError, ToolRetryError

if TYPE_CHECKING:
    from langchain.tools import StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.tool import Tool

MAX_RETRIES = 4


class NoVerifiedArgsError(Exception):
    """Raised when verified args are expected but not present."""


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
                content="You are very powerful assistant, but don't know current events."
                "Make sure you explain where you are getting the values from "
                "(e.g. from the given context, from past messages, etc.)\n",
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:"
                "\n{context}\n"
                "You will need to achieve the following goal: {input}\n"
                "The system will have a tool available, called {tool_name}.\n"
                "The format of the arguments for the tool are:\n{tool_args}\n"
                "More about the tool: {tool_description}\n"
                "\n\n----------\n\n"
                "Please provide the arguments for the tool. Prefer values in clarifications over\n"
                "those in the initial input. If there are optional args \n"
                "you can leave them empty. Do not provide placeholder arguments like \n"
                "example@example.com. Instead mark them an invalid.\n",
            ),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: ComplexLanggraphAgent) -> None:
        """Initialize the model."""
        self.llm = llm
        self.context = context
        self.agent = agent

    def invoke(self, _: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        model = self.llm.with_structured_output(ToolInputs)
        response = model.invoke(
            self.arg_parser_prompt.format_messages(
                context=self.context,
                input=self.agent.description,
                tool_name=self.agent.tool.name,
                tool_args=self.agent.tool.args_schema.model_json_schema()["properties"],
                tool_description=self.agent.tool.description,
            ),
        )
        response = ToolInputs.model_validate(response)
        return {"messages": [response.model_dump_json(indent=2)]}


class VerifierModel:
    """Model to verify the arguments for a tool."""

    arg_verifier_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an expert reviewer, trying to make sure the arguments given "
                "are correct.\n"
                "Please make sure you label if an argument was made up."
                "Do not just trust the explanations provided."
                "However if a field is marked as invalid it is likely wrong."
                "We really care if the value of an argument is not in the context, a handled "
                "clarification or goal at all (then made_up should be TRUE), but it is ok if "
                "it is there but in a different format (then made_up should be FALSE). "
                "Arguments where the value comes from a clarification should be marked as FALSE.",
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:"
                "\n{context}\n"
                "You will need to achieve the following goal: {input}\n"
                "\n\n----------\n\n"
                "Label of the following arguments as made up or not: {arguments}\n",
            ),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: ComplexLanggraphAgent) -> None:
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
                input=self.agent.description,
                arguments=tool_args,
            ),
        )
        response = VerifiedToolInputs.model_validate(response)
        self.agent.verified_args = response
        return {"messages": [response.model_dump_json(indent=2)]}


class ErrorClarifier:
    """Model to raise clarifications if we hit a soft error that the LLM believes can be solved."""

    error_clarification_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an expert reviewer, reviewing errors from previous steps in \n"
                "an Agentic flow. Your job is to review the error that was raised from the tool,"
                "and decide whether the user can help fix the error by clarifying an argument.\n"
                "A clarification is a question that will be presented back to the user to answer.\n"
                "If a clarification is useful to fix the error return the clarification"
                "including the reason the clarification is needed.\n"
                "Only ask for clarifications on Arguments to tools.\n"
                "If you believe that you can change the tool arguments to fix the error without a\n"
                "clarification we'd prefer not to ask the user.",
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:"
                "\n{context}\n"
                "The system was trying to achieve the following goal: {input}\n"
                "\n\n----------\n\n"
                "Context from previous attempts: {messages}\n",
            ),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: ComplexLanggraphAgent) -> None:
        """Initialize the model."""
        self.llm = llm
        self.context = context
        self.agent = agent

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        messages = state["messages"]
        model = self.llm.with_structured_output(InputClarification)
        response = model.invoke(
            self.error_clarification_prompt.format_messages(
                context=self.context,
                input=self.agent.description,
                messages=messages,
            ),
        )
        response = InputClarification.model_validate(response)
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
        agent: ComplexLanggraphAgent,
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
            raise NoVerifiedArgsError
        # handle any clarifications before calling
        if self.agent and self.agent.clarifications:
            for arg in verified_args.args:
                matching_clarification = next(
                    (
                        clarification
                        for clarification in self.agent.clarifications
                        if isinstance(clarification, ArgumentClarification)
                        and clarification.argument == arg.name
                    ),
                    None,
                )
                if matching_clarification and arg.value != matching_clarification.response:
                    arg.value = matching_clarification.response
                    arg.made_up = False

        model = self.llm.bind_tools(self.tools)

        messages = state["messages"]
        past_errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        response = model.invoke(
            self.tool_calling_prompt.format_messages(
                verified_args=verified_args.model_dump_json(indent=2),
                input=self.agent.description,
                past_errors=past_errors,
            ),
        )

        return {"messages": [response]}


class ComplexLanggraphAgent(BaseAgent):
    """Agent responsible for achieving a task by using langgraph.

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
        description: str,
        inputs: list[Variable],
        tool: Tool,
        clarifications: list[Clarification] | None = None,
        system_context: list[str] | None = None,
    ) -> None:
        """Initialize the agent."""
        super().__init__(description, inputs, clarifications, tool, system_context)
        self.tool = tool
        self.verified_args: VerifiedToolInputs | None = None
        self.new_clarifications: list[Clarification] = []

    @staticmethod
    def error_clarifier_or_finish(
        state: MessagesState,
    ) -> Literal["tool_agent", "error_clarifier", END]:  # type: ignore  # noqa: PGH003
        """Determine if we should attempt to handle an error internally before finishing."""
        messages = state["messages"]
        last_message = messages[-1]
        errors = [msg for msg in messages if "ToolSoftError" in msg.content]

        if "ToolSoftError" in last_message.content:
            # try internal retries first
            if len(errors) < MAX_RETRIES:
                return "tool_agent"

            # try to raise a clarification
            return "error_clarifier"
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
            # check if there's a clarification that matches the arg + value
            matching_clarification = next(
                (
                    clarification
                    for clarification in self.clarifications or []
                    if clarification.resolved
                    and getattr(clarification, "argument", None) == arg.name
                    and clarification.response == arg.value
                ),
                None,
            )
            # args shouldn't be marked as made up if they're clarified
            if not matching_clarification:
                self.new_clarifications.append(
                    InputClarification(
                        argument=arg.name,
                        user_guidance=f"Missing Argument: {arg.name}",
                    ),
                )
        if len(self.new_clarifications) > 0:
            return END

        state.update({"messages": [arguments.model_dump_json(indent=2)]})  # type: ignore  # noqa: PGH003
        return "tool_agent"

    def error_clarifications_or_continue(self, state: MessagesState) -> Literal["tool_agent", END]:  # type: ignore  # noqa: PGH003
        """Determine if we should raise a clarification based on the response."""
        messages = state["messages"]
        for message in messages:
            try:
                clarification = InputClarification.model_validate_json(str(message.content))
                self.new_clarifications.append(clarification)
            except ValidationError:  # noqa: PERF203
                continue
        return END

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
        if "ToolSoftError" in last_message.content:
            raise ToolRetryError(last_message.content)
        if "ToolHardError" in last_message.content:
            raise ToolFailedError(last_message.content)
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

        return Output(value="Requested clarification or failed to call tool.")

    def execute_sync(self, llm: BaseChatModel, step_outputs: dict[str, Output]) -> Output:
        """Run the core execution logic of the task."""
        if not self.tool:
            single_tool_agent = ToolLessAgent(
                self.description,
                self.inputs,
                clarifications=self.clarifications,
                system_context=self.system_context,
            )
            return single_tool_agent.execute_sync(llm, step_outputs)

        context = self._get_context(step_outputs)

        tools = [self.tool.to_langchain(return_artifact=True)]
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
            error_clarifier(error_clarifier)
            __end__([<p>__end__</p>]):::last
            __start__ --> argument_parser;
            argument_parser --> argument_verifier;
            argument_verifier -.-> tool_agent;
            argument_verifier -.-> __end__;
            tool_agent -.-> tools;
            tool_agent -.-> __end__;
            tools -.-> tool_agent;
            tools -.-> error_clarifier;
            tools -.-> __end__;
            error_clarifier -.-> tool_agent;
            error_clarifier -.-> __end__;
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
            ComplexLanggraphAgent.error_clarifier_or_finish,
        )

        workflow.add_node("error_clarifier", ErrorClarifier(llm, context, self).invoke)

        workflow.add_conditional_edges("error_clarifier", self.error_clarifications_or_continue)

        app = workflow.compile()

        invocation_result = app.invoke({"messages": []})
        return self.process_output(invocation_result["messages"][-1])
