"""A simple OneShotAgent optimized for simple tool calling tasks.

This agent invokes the OneShotToolCallingModel up to four times, but each individual
attempt is a one-shot call. It is useful when the tool call is simple, minimizing cost.
However, for more complex tool calls, the DefaultExecutionAgent is recommended as it will
be more successful than the OneShotAgent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from portia import logger
from portia.clarification import ActionClarification
from portia.config import Config
from portia.end_user import EndUser
from portia.errors import InvalidAgentError
from portia.execution_agents.execution_utils import (
    AgentNode,
)
from portia.execution_agents.output import LocalDataValue
from portia.milestone_plan import Milestone, MilestonePlan, MilestonePlanBuilder
from portia.open_source_tools.browser_tool import BrowserTool
from portia.plan import Plan, PlanBuilder
from portia.plan_run import PlanRun
from portia.tool import PortiaRemoteTool, Tool, ToolRunContext
from portia.tool_registry import DefaultToolRegistry, ToolRegistry

if TYPE_CHECKING:
    from langchain.tools import StructuredTool

    from portia.execution_agents.output import Output
    from portia.milestone_plan import Milestone
    from portia.model import GenerativeModel
    from portia.plan import Plan


dotenv.load_dotenv()


class ExitTool(Tool):
    """Tool that exits the milestone agent."""

    id: str = "exit"
    name: str = "exit"
    description: str = (
        "Use this tool to exit the milestone agent. This is useful "
        "if you have completed the milestone, or if you have reached a dead end."
    )
    output_schema: tuple[str, str] = ("None", "No output")

    def run(self, ctx: ToolRunContext, **kwargs: Any) -> None:
        """Exit the milestone agent."""
        raise NotImplementedError("This tool is not implemented.")


class PreviousMilestoneOutputTool(Tool):
    """Tool that returns the output of a previous milestone."""

    id: str
    name: str
    description: str
    output_schema: tuple[str, str] = ("None", "The output of the milestone")
    content: str

    def run(self, ctx: ToolRunContext) -> str:  # noqa: ARG002
        """Return the output of the previous milestone."""
        return self.content


class MilestoneExecutionState(MessagesState):
    """State for the execution agent."""

    plan: str


class PlanNode:
    """Node that generates a high-level bullet-point plan for the milestone."""

    plan_prompt = ChatPromptTemplate.from_messages(
        [
            # SYSTEM_PROMPT - give generic instructions on how to plan.
            SystemMessagePromptTemplate.from_template(
                (
                    "You are a professional task planner. Your task is to break the given "
                    "milestone objective into a concise, high-level plan. The plan consist of "
                    "a high level summary presented in clear, concise language, followed by a "
                    "series of bullet points for the likely steps needed (each bullet on a new line starting with '-'). "  # noqa: E501
                ),
                template_format="jinja2",
            ),
            # MAIN_PROMPT - provide the milestone task to plan for.
            HumanMessagePromptTemplate.from_template(
                (
                    "Milestone objective:\n"
                    "<milestone_task>\n{{ milestone_task }}\n</milestone_task>\n\n"
                    "Previous milestone outputs:\n"
                    "<previous_milestone_outputs>\n"
                    "{% for milestone_name, (milestone_task, output) in previous_milestone_outputs.items() %}"  # noqa: E501
                    '<previous_milestone name="{{ milestone_name }}" task="{{ milestone_task }}">\n'
                    "<milestone_output>\n{{ output }}\n</milestone_output>\n"
                    "</previous_milestone>\n"
                    "{% endfor %}\n"
                    "</previous_milestone_outputs>\n\n"
                    "Available tools to achieve this milestone:\n"
                    "<tools>\n"
                    "{% for tool in tools %}"
                    "<tool>\n"
                    "<tool_id>{{ tool.id }}</tool_id>\n"
                    "<tool_name>{{ tool.name }}</tool_name>\n"
                    "<tool_description>{{ tool.description }}</tool_description>\n"
                    "</tool>\n"
                    "{% endfor %}\n"
                    "</tools>\n\n"
                    "As context, the current UTC date is {{ current_date }}. The current UTC time is {{ current_time }}."  # noqa: E501
                    "Create the high-level plan now."
                ),
                template_format="jinja2",
            ),
        ]
    )

    def __init__(
        self,
        milestone: Milestone,
        model: GenerativeModel,
        tools: list[Tool],
        previous_milestone_outputs: dict[str, tuple[str, str]],
    ) -> None:
        """Initialize PlanNode."""
        self.milestone = milestone
        self.model = model
        self.tools = tools
        self.previous_milestone_outputs = previous_milestone_outputs

    def invoke(self, _state: MilestoneExecutionState) -> dict[str, Any]:
        """Return the bullet-point plan as a single AI message."""
        model = self.model.to_langchain()
        formatted_messages = self.plan_prompt.format_messages(
            milestone_task=self.milestone.task,
            tools=self.tools,
            previous_milestone_outputs=self.previous_milestone_outputs,
            current_date=datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            current_time=datetime.now(tz=UTC).strftime("%H:%M:%S"),
        )
        response = model.invoke(formatted_messages)
        logger().info(f"📋 Plan: {response.text()}")
        return {"messages": [response], "plan": response.text()}


class SelectToolNode:
    """Node for selecting a tool to call."""

    select_tool_prompt = ChatPromptTemplate.from_messages(
        [
            # SYSTEM_PROMPT: generic guidance for a ReAct-style tool-selection loop
            SystemMessagePromptTemplate.from_template(
                (
                    "You are an autonomous reasoning-and-acting (ReAct) agent operating inside "
                    "a loop. In each turn your ONLY responsibility is to choose exactly one tool "
                    "(and its arguments) from the list provided by the system that will best "
                    "progress the user toward their goal. Think about what you already know and "
                    "what information or action is required next. If the goal has already been "
                    "met or no tool is helpful, call the special `exit` tool. "
                    "Output ONLY a valid tool call in the format expected by the system—"
                    "do not output any additional text or explanation."
                ),
                template_format="jinja2",
            ),
            # MAIN_PROMPT: milestone-specific context for tool selection
            HumanMessagePromptTemplate.from_template(
                (
                    "Current milestone objective:\n"
                    "{{ milestone_task }}\n\n"
                    "The high level plan for the milestone is:\n"
                    "<plan>\n{{ initial_plan }}\n</plan>\n\n"
                    "The conversation so far is:\n"
                    "<conversation>\n{{ conversation }}\n</conversation>\n\n"
                    "Given the conversation so far, select the single best next tool to call "
                    "(and its arguments) that will most effectively progress toward completing "
                    "this milestone. Think step-by-step before deciding, but return ONLY the "
                    "tool call as your response."
                ),
                template_format="jinja2",
            ),
        ],
    )

    def __init__(
        self,
        model: GenerativeModel,
        tools: list[StructuredTool],
        milestone: Milestone,
    ) -> None:
        """Initialize the SelectToolNode."""
        self.model = model
        self.tools = tools
        self.milestone = milestone

    def invoke(self, state: MilestoneExecutionState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        model = self.model.to_langchain().bind_tools(self.tools)
        prior_messages: list[BaseMessage] = list(state.get("messages", []))  # type: ignore[arg-type]
        conversation_text = get_conversation_text(prior_messages, include_last_message=True)
        # past_errors = [str(msg) for msg in messages if "Error: ToolSoftError" in msg.content]
        formatted_messages = self.select_tool_prompt.format_messages(
            initial_plan=state.get("plan", ""),
            milestone_task=self.milestone.task,
            conversation=conversation_text,
        )
        response = model.invoke(formatted_messages)
        if isinstance(response, AIMessage) and response.tool_calls:
            logger().info(f"🛠️ Calling tool: {response.tool_calls[0]['name']}")
        elif isinstance(response, AIMessage) and not response.tool_calls:
            logger().info("👋 No tool calls, exiting")
        else:
            logger().info(f"❌ Unexpected response: {response}")
        return {"messages": [response]}


class ReasonNode:
    """Node that produces chain-of-thought reasoning about progress and next options."""

    reason_prompt = ChatPromptTemplate.from_messages(
        [
            # SYSTEM_PROMPT - explain the reasoning role.
            SystemMessagePromptTemplate.from_template(
                (
                    "You are an autonomous agent working towards a milestone. Your role is to "
                    "think step-by-step about the current progress towards the milestone, what "
                    "is still missing, and what actions can be taken to get closer to the milestone. "
                    "If you think the goal has been achieved, explain why. "
                    "Output your chain-of-thought reasoning in plain text. Be clear and concise. "
                    "Do NOT select a tool here - just reason about options."
                ),
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                (
                    "Milestone objective:\n"
                    "{{ milestone_task }}\n"
                    "The high level plan for the milestone is:\n"
                    "<plan>\n{{ initial_plan }}\n</plan>\n\n"
                    "Summary of the conversation so far (for context):\n<conversation>\n{{ conversation }}\n</conversation>\n\n"  # noqa: E501
                    "Available tools for the next step to select from:\n"
                    "{% for tool in tools %}"
                    "<tool>\n"
                    "<tool_id>{{ tool.id }}</tool_id>\n"
                    "<tool_name>{{ tool.name }}</tool_name>\n"
                    "<tool_description>{{ tool.description }}</tool_description>\n"
                    "</tool>\n"
                    "{% endfor %}\n"
                    "Provide your reasoning."
                ),
                template_format="jinja2",
            ),
        ]
    )

    def __init__(
        self,
        milestone: Milestone,
        model: GenerativeModel,
        tools: list[Tool],  # Unused for now but kept for consistency / future extension.
    ) -> None:
        """Initialize ReasonNode."""
        self.milestone = milestone
        self.model = model
        self.tools = tools

    def invoke(self, state: MilestoneExecutionState) -> dict[str, Any]:
        """Return a chain-of-thought AI message based on current state messages."""
        # Combine previous messages into a short textual summary for context.
        prior_messages: list[BaseMessage] = list(state.get("messages", []))  # type: ignore[arg-type]
        conversation_text = get_conversation_text(prior_messages)

        model = self.model.to_langchain()
        formatted_messages = self.reason_prompt.format_messages(
            initial_plan=state.get("plan", ""),
            milestone_task=self.milestone.task,
            conversation=conversation_text,
            tools=self.tools,
        )
        response = model.invoke(formatted_messages)
        logger().info(f"💭 Reasoning: {response.text()}")
        return {"messages": [response]}


class SummarizerNode:
    """Node that produces a concise summary of progress once milestone is complete."""

    summarizer_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                (
                    "You are a presentation agent. Your job is to read the outputs of the various "
                    "tools that have been called to achieve the current milestone and write up the "
                    "final output in a way that can be used in the downstream milestones. Your final "  # noqa: E501
                    "output should be a comprehensive write-up that satisfies the milestone objective."  # noqa: E501
                ),
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                (
                    "The milestone objective was:\n"
                    "{{ milestone_task }}\n\n"
                    "The initial plan to achieve the milestone was:\n<plan>\n{{ initial_plan }}\n</plan>\n\n"  # noqa: E501
                    "The conversation so far including all tool outputs was:\n<conversation>\n{{ conversation }}\n</conversation>\n\n"  # noqa: E501
                    "Present the final output now."
                ),
                template_format="jinja2",
            ),
        ]
    )

    def __init__(
        self,
        model: GenerativeModel,
        milestone: Milestone,
    ) -> None:
        """Initialize SummarizerNode."""
        self.model = model  # Already bound outside.
        self.milestone = milestone

    def invoke(self, state: MilestoneExecutionState) -> dict[str, Any]:
        """Generate a concise summary message based on the accumulated conversation."""
        # Prepare conversation context
        prior_messages: list[BaseMessage] = list(state.get("messages", []))  # type: ignore[arg-type]
        conversation_text = get_conversation_text(prior_messages)

        lang_model = self.model.to_langchain()
        formatted_messages = self.summarizer_prompt.format_messages(
            initial_plan=state.get("plan", ""),
            milestone_task=self.milestone.task,
            conversation=conversation_text,
        )
        response = lang_model.invoke(formatted_messages)
        logger().info(f"📝 Summarizer: {response.text()}")
        return {"messages": [response]}


def get_conversation_text(messages: list[BaseMessage], include_last_message: bool = False) -> str:
    """Get the conversation text from the messages."""
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


class MilestoneAgent:
    """Agent responsible for achieving a task by using langgraph.

    This agent performs the following steps:
    1. Extracts inputs from agent memory (if applicable)
    2. Calls the tool with unverified arguments.
    3. Retries tool calls up to 4 times.

    Methods:
        execute_sync(): Executes the core logic of the agent's task, using the provided tool

    """

    def __init__(
        self,
        milestone: Milestone,
        config: Config,
        end_user: EndUser,
        tools: list[Tool],
        plan_run: PlanRun,  # not used but required for tool_run_ctx
        plan: Plan,  # not used but required for tool_run_ctx
        previous_milestone_outputs: dict[str, tuple[str, str]],
    ) -> None:
        """Initialize the MilestoneAgent."""
        self.milestone = milestone
        self.config = config
        self.end_user = end_user
        self.tools = [*tools, ExitTool()]
        self.plan_run = plan_run
        self.plan = plan
        self.previous_milestone_outputs = previous_milestone_outputs

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task.

        This method will invoke the tool with arguments

        Returns:
            Output: The result of the agent's execution, containing the tool call result.

        """
        if not self.tools:
            raise InvalidAgentError("No tool available")

        tool_run_ctx = ToolRunContext(
            end_user=self.end_user,
            plan_run=self.plan_run,
            plan=self.plan,
            config=self.config,
            clarifications=self.plan_run.get_clarifications_for_step(),
        )

        model = self.config.get_execution_model()
        tools = [
            tool.to_langchain_with_artifact(
                ctx=tool_run_ctx,
            )
            for tool in self.tools
        ]
        tool_node = ToolNode(tools)

        graph = StateGraph(MilestoneExecutionState)
        graph.add_node(
            AgentNode.PLAN_AGENT,
            PlanNode(
                milestone=self.milestone,
                model=model,
                tools=self.tools,
                previous_milestone_outputs=self.previous_milestone_outputs,
            ).invoke,
        )
        graph.add_node(AgentNode.REASON, ReasonNode(self.milestone, model, self.tools).invoke)
        graph.add_node(
            AgentNode.SELECT_TOOL,
            SelectToolNode(
                model=model,
                tools=tools,
                milestone=self.milestone,
            ).invoke,
        )
        graph.add_node(AgentNode.TOOL_CALL, tool_node)
        graph.add_node(
            AgentNode.SUMMARIZER,
            SummarizerNode(
                model=model,
                milestone=self.milestone,
            ).invoke,
        )
        graph.add_node(
            "log_tool_response",
            log_tool_response,
        )

        graph.add_edge(START, AgentNode.PLAN_AGENT)
        graph.add_edge(AgentNode.PLAN_AGENT, AgentNode.REASON)
        graph.add_edge(AgentNode.REASON, AgentNode.SELECT_TOOL)
        graph.add_conditional_edges(
            AgentNode.SELECT_TOOL,
            tool_call_or_summarise,  # TOOL_CALL or SUMMARIZER
        )
        graph.add_edge(
            AgentNode.TOOL_CALL,
            "log_tool_response",
        )
        graph.add_edge("log_tool_response", AgentNode.REASON)
        graph.add_edge(AgentNode.SUMMARIZER, END)

        app = graph.compile()
        invocation_result = app.invoke({"messages": []}, config={"recursion_limit": 50})

        return process_output(invocation_result["messages"])


def process_output(messages: list[BaseMessage]) -> Output:
    """Process the output of the agent."""
    return LocalDataValue(
        summary=str(messages[-1].content) if messages[-1].content else "",
        value=messages[-1].content,
    )


def log_tool_response(state: MessagesState) -> dict[str, Any]:
    """Log the tool response."""
    if isinstance(state["messages"][-1], ToolMessage):
        logger().info(f"🛠️ Tool response: {state['messages'][-1].text()}")
    return {}


def tool_call_or_summarise(
    state: MessagesState,
) -> Literal[AgentNode.TOOL_CALL, AgentNode.SUMMARIZER]:
    """Decide whether to continue with another tool call or transition to summarisation."""
    if (
        isinstance(state["messages"][-1], AIMessage)
        and state["messages"][-1].tool_calls
        and state["messages"][-1].tool_calls[0]["name"] == "exit"
    ):
        return AgentNode.SUMMARIZER
    return AgentNode.TOOL_CALL


class Runner:
    """Runner for the milestone agent."""

    def __init__(self, config: Config, end_user: EndUser, tools: list[Tool]) -> None:
        """Initialize the Runner."""
        self.config = config
        self.end_user = end_user
        self.tools = tools

    def run(self, milestone_plan: MilestonePlan) -> Any:
        """Run the milestone agent."""
        plan = PlanBuilder().build()
        plan_run = PlanRun(plan_id=plan.id, end_user_id=self.end_user.external_id)
        all_allowed_prefixes = {
            prefix
            for milestone in milestone_plan.milestones
            for prefix in milestone.allowed_tool_prefixes
        }
        all_allowed_portia_tools = {
            tool.id
            for tool in self.tools
            if tool.id.startswith("portia:") and tool.id.startswith(tuple(all_allowed_prefixes))
        }

        ready_response = PortiaRemoteTool.batch_ready_check(
            self.config,
            all_allowed_portia_tools,
            ToolRunContext(
                end_user=self.end_user,
                plan_run=plan_run,
                plan=plan,
                config=self.config,
                clarifications=[],
            ),
        )
        if not ready_response.ready:
            logger().info("Tools need authorisation:")
            for clarification in ready_response.clarifications:
                logger().info(
                    f"  {clarification.user_guidance}: {clarification.action_url if isinstance(clarification, ActionClarification) else ''}"
                )
            input("Press Enter to continue...")

        prev_milestone_outputs = {}
        prev_milestone_tools = []
        for milestone in milestone_plan.milestones:
            logger().info(f"🎯 Running milestone: {milestone.name}")
            agent = MilestoneAgent(
                milestone=milestone,
                config=self.config,
                end_user=self.end_user,
                tools=[
                    tool
                    for tool in self.tools
                    if tool.id.startswith(tuple(milestone.allowed_tool_prefixes))
                ]
                + prev_milestone_tools,
                plan_run=plan_run,
                plan=plan,
                previous_milestone_outputs=prev_milestone_outputs,
            )
            result = agent.execute_sync()
            prev_milestone_outputs[milestone.name] = (milestone.task, result.get_value())
            prev_milestone_tools.append(
                PreviousMilestoneOutputTool(
                    id=f"previous_milestone_output_{milestone.name}",
                    name=milestone.name,
                    description=f"Call to get the result of the previous milestone task <task>{milestone.task}</task>",  # noqa: E501
                    content=str(result.get_value()),
                )
            )
        return result.get_value()


if __name__ == "__main__":
    milestone_plan = (
        MilestonePlanBuilder()
        .milestone(
            name="download_doc",
            task="Download the 'Portia Evals' doc from Google Drive and extract the Work Items section",
            allowed_tool_prefixes=["portia:google:"],
        )
        .milestone(
            name="add_tickets",
            task="""Convert the bullet points in the Work Items section into Linear tickets

and add them to the Linear project `Evals and Prompts`.

NB the team ID is 0d6ebd77-2755-4bf2-a654-13e252e61ac6
""",
            allowed_tool_prefixes=["portia:mcp:mcp.linear.app:"],
        )
        .starting_milestone("download_doc")
        .build()
    )
    config = Config.from_default(default_model="openai/gpt-4.1")
    tool_registry = DefaultToolRegistry(config=config) + ToolRegistry(tools=[BrowserTool()])
    end_user = EndUser(external_id="test")
    runner = Runner(config=config, end_user=end_user, tools=tool_registry.get_tools())
