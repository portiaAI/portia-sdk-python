"""The Default execution agent for hardest problems.

This agent uses multiple models (verifier, parser etc) to achieve the highest accuracy
in completing tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portia.clarification import Clarification, InputClarification
from portia.errors import InvalidAgentError, InvalidPlanRunStateError
from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.context import StepInput  # noqa: TC001
from portia.execution_agents.execution_utils import (
    MAX_RETRIES,
    AgentNode,
    get_arg_value_with_templating,
    process_output,
    template_in_required_inputs,
    tool_call_or_end,
)
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.logger import logger
from portia.model import GenerativeModel, Message
from portia.plan import Plan, ReadOnlyStep
from portia.plan_run import PlanRun, ReadOnlyPlanRun
from portia.telemetry.views import ExecutionAgentUsageTelemetryEvent, ToolCallTelemetryEvent
from portia.tool import Tool, ToolRunContext

if TYPE_CHECKING:
    from langchain.tools import StructuredTool

    from portia.config import Config
    from portia.end_user import EndUser
    from portia.execution_agents.output import Output
    from portia.execution_hooks import ExecutionHooks
    from portia.storage import AgentMemory


class ExecutionState(MessagesState):
    """State for the execution agent."""

    step_inputs: list[StepInput]


class ToolArgument(BaseModel):
    """Represents an argument for a tool as extracted from the goal and context.

    Attributes:
        name (str): The name of the argument, as requested by the tool.
        explanation (str): Explanation of the source for the value of the argument.
        value (Any | None): The value of the argument, as provided in the goal or context.
        valid (bool): Whether the value is a valid type and/or format for the given argument.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the argument, as requested by the tool.")
    explanation: str = Field(
        description="Explanation of the source for the value of the argument. "
        "For large arguments, includes templating was or wasn't used.",
    )
    value: Any | None = Field(
        description="Value of the argument, as provided by in the goal or context.",
    )
    valid: bool = Field(
        description="Whether the value is a valid type and or format for the given argument.",
    )


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
    explanation: str | None = Field(
        default=None,
        description="The reason the value was judged to be made up or not.",
    )

    schema_invalid: bool = Field(
        default=False,
        description="Whether the pydantic schema is invalid or not for this arg.",
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
        model (Model): The language model used for argument parsing.
        context (str): The context for argument generation.
        agent (DefaultExecutionAgent): The agent using the parser model.

    Attributes:
        arg_parser_prompt (ChatPromptTemplate): The prompt template for argument parsing.
        model (Model): The language model used.
        context (str): The context for argument generation.
        agent (DefaultExecutionAgent): The agent using the parser model.
        previous_errors (list[str]): A list of previous errors encountered during parsing.
        retries (int): The number of retries attempted for parsing.

    """

    arg_parser_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are a highly capable assistant tasked with generating valid arguments for "
                "tools based on provided input. "
                "While you are not aware of current events, you excel at reasoning "
                "and adhering to instructions. "
                "Your responses must clearly explain the source of each argument "
                "(e.g., context, past messages, clarifications). "
                "Avoid assumptions or fabricated information. "
                "Pay attention to previous errors given and do not repeat them. "
                "If any of the inputs is a large string and you want to use it verbatim, rather "
                "than repeating it, you should provide the name in curly braces and it will be "
                "templated in before the tool is called. "
                "For example, if you wish to use an input called '$large_input_value' verbatim, "
                "you should enter '{{{{$large_input_value}}}}' (double curly braces "
                "and include the $ in the name) and the value will be templated in before the tool "
                "is called. If there is a $ in the variable name, MAKE SURE you keep it. You "
                "should definitely use this templating for any input values over 1000 words that "
                "you want to use verbatim.",
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:\n{context}\n"
                "Task: {task}\n"
                "The system has a tool available named '{tool_name}'.\n"
                "Argument schema for the tool:\n{tool_args}\n"
                "Description of the tool: {tool_description}\n"
                "\n\n----------\n\n"
                "The following section contains previous errors. "
                "Ensure your response avoids these errors. "
                "The one exception to this is not providing a value for a required argument. "
                "If a value cannot be extracted from the context, you can leave it blank. "
                "Do not assume a default value that meets the type expectation or is a common testing value. "  # noqa: E501
                "Here are the previous errors:\n"
                "{previous_errors}\n"
                "\n\n----------\n\n"
                "Please provide the arguments for the tool. Adhere to the following guidelines:\n"
                "- If a tool needs to be called many times, you can repeat the argument\n"
                "- You may take values from the task, inputs, previous steps or clarifications\n"
                "- Prefer values clarified in follow-up inputs over initial inputs.\n"
                "- Do not provide placeholder values (e.g., 'example@example.com').\n"
                "- Do not include references to any of the input values (e.g. 'as provided in "
                "the input'): you must put the exact value the tool should be called with in "
                "the value field\n"
                "- If the tool has a list return it as a list, i.e. [1, 2, 3] not '[1, 2, 3]'"
                "- Ensure arguments align with the tool's schema and intended use.\n\n"
                "You must return the arguments in the following JSON format:\n"
                "- If any of the inputs is a large string and you want to use it verbatim, rather "
                "than repeating it, you should provide the name in curly braces and it will be "
                "templated in before the tool is called. For example, if you wish to use an input "
                "called '$large_input_value' verbatim, you should enter "
                "'{{{{$large_input_value}}}}' "
                "(double curly braces and include the $ in the name) and the value will be "
                "templated in before the tool is called.  You should definitely use this "
                "templating for any input values over 1000 words that you want to use verbatim.\n"
                "- If you use templating, MAKE SURE to keep a $ at the start of the name if there "
                "is one and MAKE SURE to use two curly braces (not one or three) to open and close "
                "the templating.\n"
                "- Generate the fields in the order they appear in the classes below."
                "class ToolInputs:\n"
                "  args: List[ToolArgument]  # List of tool arguments.\n\n"
                "class ToolArgument:\n"
                "  name: str  # Name of the argument requested by the tool.\n"
                "  explanation: str  # Explanation of the source for the value of the argument. "
                "For large arguments, include why you did or didn't use templating.\n"
                "  value: Any | None  # Value of the argument from the goal or context.\n"
                "  valid: bool  # Whether the value is valid for the argument.\n\n",
            ),
        ],
    )

    def __init__(
        self,
        model: GenerativeModel,
        agent: DefaultExecutionAgent,
        tool_context: ToolRunContext,
    ) -> None:
        """Initialize the model.

        Args:
            model (Model): The language model used for argument parsing.
            agent (DefaultExecutionAgent): The agent using the parser model.
            tool_context (ToolRunContext): The context for the tool.

        """
        self.model = model
        self.agent = agent
        self.tool_context = tool_context
        self.previous_errors: list[str] = []
        self.retries = 0

    def invoke(self, state: ExecutionState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Args:
            state (ExecutionState): The current state of the conversation.

        Returns:
            dict[str, Any]: The response after invoking the model.

        Raises:
            InvalidRunStateError: If the agent's tool is not available.

        """
        if not self.agent.tool:
            raise InvalidPlanRunStateError("Parser model has no tool")

        formatted_messages = self.arg_parser_prompt.format_messages(
            context=self.agent.get_system_context(self.tool_context, state["step_inputs"]),
            task=self.agent.step.task,
            tool_name=self.agent.tool.name,
            tool_args=self.agent.tool.args_json_schema(),
            tool_description=self.agent.tool.description,
            previous_errors=",".join(self.previous_errors),
        )

        errors = []
        tool_inputs: ToolInputs | None = None
        try:
            logger().trace("LLM call: argument parsing")
            response = self.model.get_structured_response(
                messages=[Message.from_langchain(m) for m in formatted_messages],
                schema=ToolInputs,
            )
            tool_inputs = ToolInputs.model_validate(response)
        except ValidationError as e:
            errors.append("Invalid JSON for ToolInputs: " + str(e) + "\n")
        else:
            test_args = {}
            for arg in tool_inputs.args:
                test_args[arg.name] = get_arg_value_with_templating(state["step_inputs"], arg.value)
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

        return {"messages": [tool_inputs.model_dump_json(indent=2)] if tool_inputs else []}


class VerifierModel:
    """A model to verify the arguments for a tool.

    This model ensures that the arguments passed to a tool are valid, determining whether they are
    "made up" or not based on the context and specific rules. The verification process uses an LLM
    to analyze the context and tool arguments and returns a structured validation output.

    Attributes:
        arg_verifier_prompt (ChatPromptTemplate): The prompt template used for arg verification.
        model (Model): The model used to invoke the verification process.
        agent (DefaultExecutionAgent): The agent responsible for handling the verification process.

    """

    arg_verifier_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an expert reviewer. Your task is to validate and label arguments "
                "provided. You must return the made_up field based "
                "on the rules below.\n - An argument is made up if we cannot tell where the value "
                "came from in the goal or context.\n- You should verify that the explanations are "
                "grounded in the goal or context before trusting them."
                "\n- If an argument is marked as invalid it is likely made up.\n- If an argument "
                "is a placeholder (e.g. 'example@example.com') it is likely made up."
                "\n- We really care if the value of an argument is not in the context, a handled "
                "clarification or goal at all (then made_up should be TRUE), but it is ok if "
                "it is there but in a different format, or if it can be reasonably derived from the"
                " information that is there (then made_up should be FALSE). "
                "\n- Arguments where the value comes from a clarification should be marked as FALSE"
                "\n- Some large inputs may be provided as a template variable (e.g. "
                "'{{{{$large_input_value}}}}'). This is fine and the value will be "
                "templated in before the tool is called.\n"
                "The output must conform to the following schema:\n\n"
                "class VerifiedToolArgument:\n"
                "  name: str  # Name of the argument requested by the tool.\n"
                "  value: Any | None  # Value of the argument from the goal or context. "
                "USE EXACTLY the type of the argument provided in the list of arguments provided.\n"
                "  explanation: str # The reason the value was judged to be made up or not\n"
                "  made_up: bool  # if the value is made_up based on the given rules.\n\n"
                "class VerifiedToolInputs:\n"
                "  args: List[VerifiedToolArgument]  # List of tool arguments.\n\n"
                "Please ensure the output matches the VerifiedToolInputs schema.",
            ),
            HumanMessagePromptTemplate.from_template(
                "You will need to achieve the following goal: {task}\n"
                "\n\n----------\n\n"
                "Context for user input and past steps:"
                "\n{context}\n"
                "The system has a tool available named '{tool_name}' "
                "with description: {tool_description}.\n"
                "Argument schema for the tool:\n{tool_args}\n"
                "\n\n----------\n\n"
                "Label the following arguments as made up or not using the goal and context provided: {arguments}\n",  # noqa: E501
            ),
        ],
    )

    def __init__(
        self,
        model: GenerativeModel,
        agent: DefaultExecutionAgent,
        tool_context: ToolRunContext,
    ) -> None:
        """Initialize the model.

        Args:
            model (Model): The model used for argument verification.
            context (str): The context for argument generation.
            agent (DefaultExecutionAgent): The agent using the verifier model.
            tool_context (ToolRunContext): The context for the tool.

        """
        self.model = model
        self.agent = agent
        self.tool_context = tool_context

    def invoke(self, state: ExecutionState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Args:
            state (ExecutionState): The current state of the conversation.

        Returns:
            dict[str, Any]: The response after invoking the model.

        Raises:
            InvalidRunStateError: If the agent's tool is not available.

        """
        if not self.agent.tool:
            raise InvalidPlanRunStateError("Verifier model has no tool")

        messages = state["messages"]
        tool_args = messages[-1].content
        formatted_messages = self.arg_verifier_prompt.format_messages(
            context=self.agent.get_system_context(self.tool_context, state["step_inputs"]),
            task=self.agent.step.task,
            arguments=tool_args,
            tool_name=self.agent.tool.name,
            tool_args=self.agent.tool.args_json_schema(),
            tool_description=self.agent.tool.description,
        )
        logger().trace("LLM call: argument verification")
        response = self.model.get_structured_response(
            messages=[Message.from_langchain(m) for m in formatted_messages],
            schema=VerifiedToolInputs,
        )
        response = VerifiedToolInputs.model_validate(response)
        # Validate the arguments against the tool's schema
        response = self._validate_args_against_schema(response, state["step_inputs"])
        self.agent.verified_args = response

        return {"messages": [response.model_dump_json(indent=2)]}

    def _validate_args_against_schema(
        self, tool_inputs: VerifiedToolInputs, step_inputs: list[StepInput]
    ) -> VerifiedToolInputs:
        """Validate tool arguments against the tool's schema and mark invalid ones as made up.

        Args:
            tool_inputs (VerifiedToolInputs): The tool_inputs to validate against the tool schema.
            step_inputs (list[StepInput]): The step inputs to use for templating.

        Returns:
            Updated VerifiedToolInputs with invalid args marked with schema_invalid=True.

        """
        arg_dict = {arg.name: arg.value for arg in tool_inputs.args}

        try:
            if self.agent.tool:
                for arg_name, arg_value in arg_dict.items():
                    arg_dict[arg_name] = get_arg_value_with_templating(step_inputs, arg_value)
                self.agent.tool.args_schema.model_validate(arg_dict)
        except ValidationError as e:
            # Extract the arg names from the pydantic error to mark them as schema_invalid = True.
            # At this point we know the arguments are invalid, so we can trigger a clarification
            # request.
            invalid_arg_names = set()
            for error in e.errors():
                # Gemini often returns lists as '[1,2,3]', but downstream LLMs can handle this.
                if error["msg"] == "Input should be a valid list" and error["input"].startswith(
                    "["
                ):
                    continue
                if error.get("loc") and len(error["loc"]) > 0:
                    invalid_arg_names.add(error["loc"][0])

            [
                setattr(arg, "schema_invalid", True)
                for arg in tool_inputs.args
                if arg.name in invalid_arg_names
            ]
        # Mark any made up arguments that are None and optional as not made up.
        # We don't need to raise a clarification for these
        [
            setattr(arg, "made_up", False)
            for arg in tool_inputs.args
            if arg.value is None
            and arg.made_up
            and self.agent.tool
            and not self.agent.tool.args_schema.model_fields[arg.name].is_required()
        ]
        return tool_inputs


class ToolCallingModel:
    """Model to call the tool with the verified arguments."""

    tool_calling_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are very powerful assistant that calls tools with the provided "
                "arguments. You don't know current events. "
                "If any values are too large to be provided to you in full, they will be provided "
                "in curly braces with a value to be templated in (e.g. "
                "'{{{{$large_output_value}}}}'). "
                "This is fine - please keep these templated values inside double curly braces and "
                "DO NOT REMOVE the leading $ on the name - for example, keep it as "
                "'{{{{$large_output_value}}}}' and not "
                "'{{{{large_output_value}}}}'. These values will then be templated in "
                "before the tool is called.\n",
            ),
            HumanMessagePromptTemplate.from_template(
                "context:\n{verified_args}\n"
                "Use the provided tool with the arguments in the context, as "
                "long as they are valid.\n"
                "Make sure you don't repeat past errors: {past_errors}\n",
            ),
        ],
    )

    def __init__(
        self,
        model: GenerativeModel,
        tools: list[StructuredTool],
        agent: DefaultExecutionAgent,
    ) -> None:
        """Initialize the model.

        Args:
            model (GenerativeModel): The language model used for argument parsing.
            agent (DefaultExecutionAgent): The agent using the parser model.
            tools (list[StructuredTool]): The tools to pass to the model.

        """
        self.model = model
        self.agent = agent
        self.tools = tools

    def invoke(self, state: ExecutionState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        Args:
            state (ExecutionState): The current state of the conversation.

        Returns:
            dict[str, Any]: The response after invoking the model.

        Raises:
            InvalidRunStateError: If the agent's tool is not available.

        """
        verified_args = self.agent.verified_args
        if not verified_args:
            raise InvalidPlanRunStateError
        # handle any clarifications before calling
        if self.agent and self.agent.plan_run.outputs.clarifications:
            for arg in verified_args.args:
                matching_clarification = self.agent.get_last_resolved_clarification(arg.name)
                if matching_clarification and arg.value != matching_clarification.response:
                    arg.value = matching_clarification.response
                    arg.made_up = False
                    arg.schema_invalid = False

        model = self.model.to_langchain().bind_tools(self.tools)

        messages = state["messages"]
        past_errors = [msg for msg in messages if "ToolSoftError" in msg.content]
        self.agent.telemetry.capture(
            ToolCallTelemetryEvent(tool_id=self.agent.tool.id if self.agent.tool else None)
        )
        logger().trace("LLM call: tool calling")
        response = model.invoke(
            self.tool_calling_prompt.format_messages(
                verified_args=verified_args.model_dump_json(indent=2),
                past_errors=past_errors,
            ),
        )
        result = template_in_required_inputs(response, state["step_inputs"])

        if (
            self.agent.execution_hooks
            and self.agent.execution_hooks.before_tool_call
            and self.agent.tool
        ):
            for tool_call in response.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
                logger().debug("Calling before_tool_call execution hook")
                clarification = self.agent.execution_hooks.before_tool_call(
                    self.agent.tool,
                    tool_call.get("args"),
                    ReadOnlyPlanRun.from_plan_run(self.agent.plan_run),
                    ReadOnlyStep.from_step(self.agent.step),
                )
                logger().debug("Finished before_tool_call execution hook")
                if clarification:
                    self.agent.new_clarifications.append(clarification)
            if self.agent.new_clarifications:
                return {"messages": []}

        return {"messages": [result]}


class DefaultExecutionAgent(BaseExecutionAgent):
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
        plan: Plan,
        plan_run: PlanRun,
        config: Config,
        agent_memory: AgentMemory,
        end_user: EndUser,
        tool: Tool | None = None,
        execution_hooks: ExecutionHooks | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            plan (Plan): The plan containing the steps.
            plan_run (PlanRun): The run that defines the task execution process.
            config (Config): The configuration settings for the agent.
            agent_memory (AgentMemory): The agent memory to be used for the task.
            end_user (EndUser): The end user for this execution
            tool (Tool | None): The tool to be used for the task (optional).
            execution_hooks (ExecutionHooks | None): The execution hooks for the agent.

        """
        super().__init__(
            plan=plan,
            plan_run=plan_run,
            config=config,
            end_user=end_user,
            agent_memory=agent_memory,
            tool=tool,
            execution_hooks=execution_hooks,
        )
        self.verified_args: VerifiedToolInputs | None = None

    def clarifications_or_continue(
        self,
        state: ExecutionState,
    ) -> Literal[AgentNode.TOOL_AGENT, END]:  # type: ignore  # noqa: PGH003
        """Determine if we should continue with the tool call or request clarifications instead.

        Args:
            state (ExecutionState): The current state of the conversation.

        Returns:
            Literal[AgentNode.TOOL_AGENT, END]: The next node we should route to.

        """
        messages = state["messages"]
        last_message = messages[-1]
        arguments = VerifiedToolInputs.model_validate_json(str(last_message.content))

        for arg in arguments.args:
            if not arg.made_up and not arg.schema_invalid:
                continue
            matching_clarification = self.get_last_resolved_clarification(arg.name)

            if not matching_clarification:
                self.new_clarifications.append(
                    InputClarification(
                        plan_run_id=self.plan_run.id,
                        argument_name=arg.name,
                        user_guidance=f"Missing Argument: {arg.name}",
                        step=self.plan_run.current_step_index,
                        source="Default execution agent",
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
        for clarification in self.plan_run.outputs.clarifications:
            if (
                clarification.resolved
                and getattr(clarification, "argument_name", None) == arg_name
                and clarification.step == self.plan_run.current_step_index
            ):
                matching_clarification = clarification
        return matching_clarification

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task.

        This method will invoke the tool with arguments that are parsed and verified first.

        Returns:
            Output: The result of the agent's execution, containing the tool call result.

        """
        self.telemetry.capture(
            ExecutionAgentUsageTelemetryEvent(
                agent_type="default",
                model=str(self.config.get_execution_model()),
                sync=True,
                tool_id=self.tool.id if self.tool else None,
            )
        )

        if not self.tool:
            raise InvalidAgentError("Tool is required for DefaultExecutionAgent")

        tool_run_ctx = ToolRunContext(
            end_user=self.end_user,
            plan_run=self.plan_run,
            plan=self.plan,
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
        """
        The execution graph represented here can be generated using
        `print(app.get_graph().draw_mermaid())` on the compiled run (and running any agent
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

        graph.add_node(AgentNode.TOOL_AGENT, ToolCallingModel(model, tools, self).invoke)
        if self.verified_args:
            graph.add_edge(START, AgentNode.TOOL_AGENT)
        else:
            graph.add_node(AgentNode.MEMORY_EXTRACTION, MemoryExtractionStep(self).invoke)
            graph.add_edge(START, AgentNode.MEMORY_EXTRACTION)
            graph.add_node(AgentNode.ARGUMENT_PARSER, ParserModel(model, self, tool_run_ctx).invoke)
            graph.add_edge(AgentNode.MEMORY_EXTRACTION, AgentNode.ARGUMENT_PARSER)
            graph.add_node(
                AgentNode.ARGUMENT_VERIFIER, VerifierModel(model, self, tool_run_ctx).invoke
            )
            graph.add_edge(AgentNode.ARGUMENT_PARSER, AgentNode.ARGUMENT_VERIFIER)
            graph.add_conditional_edges(
                AgentNode.ARGUMENT_VERIFIER,
                self.clarifications_or_continue,
            )

        graph.add_node(AgentNode.TOOLS, tool_node)
        graph.add_node(
            AgentNode.SUMMARIZER,
            StepSummarizer(self.config, model, self.tool, self.step).invoke,
        )
        graph.add_conditional_edges(
            AgentNode.TOOLS,
            lambda state: self.next_state_after_tool_call(self.config, state, self.tool),
        )
        graph.add_conditional_edges(
            AgentNode.TOOL_AGENT,
            tool_call_or_end,
        )
        graph.add_edge(AgentNode.SUMMARIZER, END)

        app = graph.compile()
        invocation_result = app.invoke({"messages": [], "step_inputs": []})
        return process_output(
            self.step,
            invocation_result["messages"],
            self.tool,
            self.new_clarifications,
        )
