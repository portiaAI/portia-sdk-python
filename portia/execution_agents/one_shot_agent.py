"""A simple OneShotAgent optimized for simple tool calling tasks.

This agent invokes the OneShotToolCallingModel up to four times, but each individual
attempt is a one-shot call. It is useful when the tool call is simple, minimizing cost.
However, for more complex tool calls, the DefaultExecutionAgent is recommended as it will
be more successful than the OneShotAgent.
"""

from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
)

from portia.config import FEATURE_FLAG_ONE_SHOT_AGENT_CLARIFICATIONS_ENABLED
from portia.errors import InvalidAgentError

from portia.execution_agents.base_execution_agent import BaseExecutionAgent
from portia.execution_agents.execution_utils import (
    AgentNode,
    template_in_required_inputs,
    process_output,
    tool_call_or_end,
)
from portia.execution_agents.memory_extraction import MemoryExtractionStep
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.execution_agents.clarification_tool import ClarificationTool
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

    arg_parser_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are a highly capable assistant tasked with calling tools based on the "
                "provided inputs. "
                "While you are not aware of current events, you excel at reasoning "
                "and adhering to instructions. "
                "Avoid assumptions or fabricated information. "
                "{% if use_clarification_tool %}If you are unsure of an argument to use for the "
                "tool, you can use the clarification tool to clarify what the argument should be. "
                "{% endif %}If any of the inputs is a large string and you want to use it "
                "verbatim, rather than repeating it, you should provide the name in curly braces "
                "to the tool call and it will be templated in before the tool is called. "
                "For example, if you wish to use an input called '$large_input_value' verbatim, "
                "you should enter '{{ '{{' }}$large_input_value{{ '}}' }}' (double curly braces "
                "and include the $ in the name) and the value will be templated in before the tool "
                "is called.  You should definitely use this templating for any input values over "
                "1000 words that you want to use verbatim.",
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                "Context for user input and past steps:\n{{ context }}\n"
                "Task: {{ task }}\n"
                "The system has a tool available named '{{ tool_name }}'.\n"
                "Argument schema for the tool:\n{{ tool_args }}\n"
                "Description of the tool: {{ tool_description }}\n"
                "{% if use_clarification_tool %}\n"
                "You also have a clarification tool available with "
                "argument schema:\n{{ clarification_tool_args }}\n"
                "{% endif %}"
                "\n\n----------\n\n"
                "The following section contains previous errors. "
                "Ensure your response avoids these errors. "
                "The one exception to this is not providing a value for a required argument. "
                "If a value cannot be extracted from the context, you can leave it blank. "
                "Do not assume a default value that meets the type expectation or is a common testing value. "  # noqa: E501
                "Here are the previous errors:\n"
                "{{ previous_errors }}\n"
                "\n\n----------\n\n"
                "Please call the tool to achieve the above task, following the guidelines below:\n"
                "- If a tool needs to be called many times, you can repeat the argument\n"
                "- You may take values from the task, inputs, previous steps or clarifications\n"
                "- Prefer values clarified in follow-up inputs over initial inputs.\n"
                "- Do not provide placeholder values (e.g., 'example@example.com').\n"
                "- Do not include references to any of the input values (e.g. 'as provided in "
                "the input'): you must put the exact value the tool should be called with in "
                "the value field\n"
                "- Ensure arguments align with the tool's schema and intended use."
                "{% if use_clarification_tool %}- If you are unsure of an argument to use for the "
                "tool, you can use the clarification tool to clarify what the argument should be.\n"
                "{% endif %}",
                template_format="jinja2",
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
        if not self.agent.tool:
            raise InvalidAgentError("Parser model has no tool")

        model = self.model.to_langchain().bind_tools(self.tools)
        messages = state["messages"]
        past_errors = [str(msg) for msg in messages if "ToolSoftError" in msg.content]
        clarification_tool = ClarificationTool(step=self.agent.plan_run.current_step_index)
        formatted_messages = self.arg_parser_prompt.format_messages(
            context=self.agent.get_system_context(self.tool_context, state["step_inputs"]),
            task=self.agent.step.task,
            tool_name=self.agent.tool.name,
            tool_args=self.agent.tool.args_json_schema(),
            tool_description=self.agent.tool.description,
            use_clarification_tool=self.agent.config.feature_flags[
                FEATURE_FLAG_ONE_SHOT_AGENT_CLARIFICATIONS_ENABLED
            ],
            clarification_tool_args=clarification_tool.args_json_schema(),
            previous_errors=",".join(past_errors),
        )
        response = model.invoke(formatted_messages)
        result = template_in_required_inputs(response, state["step_inputs"])

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
        return {"messages": [result]}


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
        clarification_tool = ClarificationTool(step=self.plan_run.current_step_index)
        if self.config.feature_flags[FEATURE_FLAG_ONE_SHOT_AGENT_CLARIFICATIONS_ENABLED]:
            tools.append(clarification_tool.to_langchain_with_artifact(ctx=tool_run_ctx))
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
