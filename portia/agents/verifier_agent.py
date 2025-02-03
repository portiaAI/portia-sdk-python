"""The Verifier Agent for hardest problems.

This agent uses multiple models (verifier, parser etc) to achieve the highest accuracy
in completing tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portia.agents.base_agent import BaseAgent, Output
from portia.agents.execution_utils import (
    MAX_RETRIES,
    AgentNode,
    next_state_after_tool_call,
    process_output,
    tool_call_or_end,
)
from portia.agents.utils.step_summarizer import StepSummarizer
from portia.clarification import Clarification, InputClarification
from portia.errors import InvalidAgentError, InvalidWorkflowStateError
from portia.execution_context import get_execution_context
from portia.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from langchain.tools import StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.config import Config
    from portia.plan import Step
    from portia.tool import Tool
    from portia.workflow import Workflow


class ToolArgument(BaseModel):
    """Represents an argument for a tool as extracted from the goal and context.

    Attributes:
        name (str): The name of the argument, as requested by the tool.
        value (Any | None): The value of the argument, as provided in the goal or context.
        valid (bool): Whether the value is a valid type and/or format for the given argument.
        explanation (str): Explanation of the source for the value of the argument.

    """

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
    """Represents the inputs for a tool.

    Attributes:
        args (list[ToolArgument]): Arguments for the tool.

    """

    args: list[ToolArgument] = Field(description="Arguments for the tool.")


class VerifiedToolArgument(BaseModel):
    """Represents an argument for a tool after being verified by an agent.

    Attributes:
        name (str): The name of the argument, as requested by the tool.
        value (Any | None): The value of the argument, as provided in the goal or context.
        made_up (bool): Whether the value was made up or not. Should be false if the value was
        provided by the user.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the argument, as requested by the tool.")
    value: Any | None = Field(
        description="Value of the argument, as provided by in the goal or context.",
    )

    # We call this "made_up" and not "hallucinated" because the latter was making OpenAI's model
    # produce invalid JSON.
    made_up: bool = Field(
        default=False,
        description="Whether the value was made up or not. "
        "Should be false if the value was provided by the user, even if in a different format."
        "User provided values can be in the context, in the goal or the result of previous steps.",
    )


class VerifiedToolInputs(BaseModel):
    """Represents the inputs for a tool after being verified by an agent.

    Attributes:
        args (list[VerifiedToolArgument]): Arguments for the tool.

    """

    args: list[VerifiedToolArgument] = Field(description="Arguments for the tool.")


class ParserModel:
    """Model to parse the arguments for a tool.

    Args:
        llm (BaseChatModel): The language model used for argument parsing.
        context (str): The context for argument generation.
        agent (VerifierAgent): The agent using the parser model.

    Attributes:
        arg_parser_prompt (ChatPromptTemplate): The prompt template for argument parsing.
        llm (BaseChatModel): The language model used.
        context (str): The context for argument generation.
        agent (VerifierAgent): The agent using the parser model.
        previous_errors (list[str]): A list of previous errors encountered during parsing.
        retries (int): The number of retries attempted for parsing.

    """

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
                "The following section contains previous errors. "
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
        """Initialize the model.

        Args:
            llm (BaseChatModel): The language model used for argument parsing.
            context (str): The context for argument generation.
            agent (VerifierAgent): The agent using the parser model.

        """
        self.llm = llm
        self.context = context
        self.agent = agent
        self.previous_errors: list[str] = []
        self.retries = 0

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Args:
            state (MessagesState): The current state of the conversation.

        Returns:
            dict[str, Any]: The response after invoking the model.

        Raises:
            InvalidWorkflowStateError: If the agent's tool is not available.

        """
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

        test_args = {}
        errors = []
        for arg in response.args:
            test_args[arg.name] = arg.value
            if not arg.valid:
                errors.append(f"Error in argument {arg.name}: {arg.explanation}\n")

        # also test the ToolInputs that have come back
        # actually work for the schema of the tool
        # if not we can retry
        try:
            self.agent.tool.args_schema.model_validate(test_args)
        except ValidationError as e:
            errors.append(str(e) + "\n")

        if errors:
            self.previous_errors.extend(errors)
            self.retries += 1
            if self.retries <= MAX_RETRIES:
                return self.invoke(state)
            # Previously we would raise an error here, but this restricts the agent from
            # being able to raise clarifications for the tool arguments marked as invalid.
            # Missing tool arguments are often represented as None, which isn't a compatible
            # type for non-optional arguments.
            #
            # Here is a Linear ticket to fix this:
            # https://linear.app/portialabs/issue/POR-456

        return {"messages": [response.model_dump_json(indent=2)]}


class VerifierModel:
    """A model to verify the arguments for a tool.

    This model ensures that the arguments passed to a tool are valid, determining whether they are
    "made up" or not based on the context and specific rules. The verification process uses an LLM
    to analyze the context and tool arguments and returns a structured validation output.

    Attributes:
        arg_verifier_prompt (ChatPromptTemplate): The prompt template used for arg verification.
        llm (BaseChatModel): The language model used to invoke the verification process.
        context (str): The context in which the tool arguments are being validated.
        agent (VerifierAgent): The agent responsible for handling the verification process.

    """

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
        """Initialize the model.

        Args:
            llm (BaseChatModel): The language model used for argument parsing.
            context (str): The context for argument generation.
            agent (VerifierAgent): The agent using the parser model.

        """
        self.llm = llm
        self.context = context
        self.agent = agent

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Args:
            state (MessagesState): The current state of the conversation.

        Returns:
            dict[str, Any]: The response after invoking the model.

        Raises:
            InvalidWorkflowStateError: If the agent's tool is not available.

        """
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

        # Validate the arguments against the tool's schema
        response = self._validate_args_against_schema(response)
        self.agent.verified_args = response

        return {"messages": [response.model_dump_json(indent=2)]}

    def _validate_args_against_schema(self, tool_inputs: VerifiedToolInputs) -> VerifiedToolInputs:
        """Validate tool arguments against the tool's schema and mark invalid ones as made up.

        Args:
            tool_inputs (VerifiedToolInputs): The tool_inputs to validate against the tool schema.

        Returns:
            Updated VerifiedToolInputs with invalid/missing args marked with made_up=True if found.

        """
        arg_dict = {arg.name: arg.value for arg in tool_inputs.args}

        try:
            if self.agent.tool:
                self.agent.tool.args_schema.model_validate(arg_dict)
        except ValidationError as e:
            # Extract the arg names from the pydantic error to mark them as made_up = True.
            # At this point we know the arguments are invalid, so we can trigger a clarification
            # request.
            invalid_arg_names = {
                error["loc"][0]
                for error in e.errors()
                if error.get("loc") and len(error["loc"]) > 0
            }
            [
                setattr(arg, "made_up", True)
                for arg in tool_inputs.args
                if arg.name in invalid_arg_names
            ]

        return tool_inputs

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
        """Initialize the model.

        Args:
            llm (BaseChatModel): The language model used for argument parsing.
            context (str): The context for argument generation.
            agent (VerifierAgent): The agent using the parser model.
            tools (list[StructuredTool]): The tools to pass to the model.

        """
        self.llm = llm
        self.context = context
        self.agent = agent
        self.tools = tools

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Args:
            state (MessagesState): The current state of the conversation.

        Returns:
            dict[str, Any]: The response after invoking the model.

        Raises:
            InvalidWorkflowStateError: If the agent's tool is not available.

        """
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
        """Initialize the agent.

        Args:
            step (Step): The current step in the task plan.
            workflow (Workflow): The workflow that defines the task execution process.
            config (Config): The configuration settings for the agent.
            tool (Tool | None): The tool to be used for the task (optional).

        """
        super().__init__(step, workflow, config, tool)
        self.verified_args: VerifiedToolInputs | None = None
        self.new_clarifications: list[Clarification] = []

    def clarifications_or_continue(
        self,
        state: MessagesState,
    ) -> Literal[AgentNode.TOOL_AGENT, END]:  # type: ignore  # noqa: PGH003
        """Determine if we should continue with the tool call or request clarifications instead.

        Args:
            state (MessagesState): The current state of the conversation.

        Returns:
            Literal[AgentNode.TOOL_AGENT, END]: The next node we should route to.

        """
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
                        step=self.workflow.current_step_index,
                    ),
                )
        if self.new_clarifications:
            return END

        state.update({"messages": [arguments.model_dump_json(indent=2)]})  # type: ignore  # noqa: PGH003
        return AgentNode.TOOL_AGENT

    def get_last_resolved_clarification(
        self,
        arg_name: str,
    ) -> Clarification | None:
        """Return the last argument clarification that matches the given arg_name.

        Args:
            arg_name (str): The name of the argument to match clarifications for

        Returns:
            Clarification | None: The matched clarification

        """
        matching_clarification = None
        for clarification in self.workflow.outputs.clarifications:
            if (
                clarification.resolved
                and getattr(clarification, "argument_name", None) == arg_name
                and clarification.step == self.workflow.current_step_index
            ):
                matching_clarification = clarification
        return matching_clarification

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task.

        This method will invoke the tool with arguments that are parsed and verified first.

        Returns:
            Output: The result of the agent's execution, containing the tool call result.

        """
        if not self.tool:
            raise InvalidAgentError("No tool available")

        context = self.get_system_context()
        llm = LLMWrapper(self.config).to_langchain()
        tools = [
            self.tool.to_langchain_with_artifact(
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
                summarizer(summarizer)
                __end__([<p>__end__</p>]):::last
                __start__ --> argument_parser;
                argument_parser --> argument_verifier;
                summarizer --> __end__;
                argument_verifier -.-> tool_agent;
                argument_verifier -.-> __end__;
                tools -.-> tool_agent;
                tools -.-> summarizer;
                tools -.-> __end__;
                tool_agent -.-> tools;
                tool_agent -.-> __end__;
                classDef default fill:#f2f0ff,line-height:1.2
                classDef first fill-opacity:0
                classDef last fill:#bfb6fc
        """

        workflow.add_node(AgentNode.TOOL_AGENT, ToolCallingModel(llm, context, tools, self).invoke)
        if self.verified_args:
            workflow.add_edge(START, AgentNode.TOOL_AGENT)
        else:
            workflow.add_node(AgentNode.ARGUMENT_PARSER, ParserModel(llm, context, self).invoke)
            workflow.add_node(AgentNode.ARGUMENT_VERIFIER, VerifierModel(llm, context, self).invoke)
            workflow.add_edge(START, AgentNode.ARGUMENT_PARSER)
            workflow.add_edge(AgentNode.ARGUMENT_PARSER, AgentNode.ARGUMENT_VERIFIER)
            workflow.add_conditional_edges(
                AgentNode.ARGUMENT_VERIFIER,
                self.clarifications_or_continue,
            )

        workflow.add_node(AgentNode.TOOLS, tool_node)
        workflow.add_node(AgentNode.SUMMARIZER, StepSummarizer(llm).invoke)
        workflow.add_conditional_edges(
            AgentNode.TOOLS,
            lambda state: next_state_after_tool_call(state, self.tool),
        )
        workflow.add_conditional_edges(
            AgentNode.TOOL_AGENT,
            tool_call_or_end,
        )
        workflow.add_edge(AgentNode.SUMMARIZER, END)

        app = workflow.compile()
        invocation_result = app.invoke({"messages": []})
        return process_output(
            invocation_result["messages"][-1],
            self.tool,
            self.new_clarifications,
        )
