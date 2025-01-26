"""Agent designed when no tool is needed.

This is useful for solving tasks where the LLM intrinsically has the knowledge or
for creative tasks. Anything that an LLM can generate itself can use the ToolLess Agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from portia.agents.base_agent import BaseAgent, Output
from portia.agents.execution_utils import AgentNode
from portia.clarification import Clarification, MultipleChoiceClarification
from portia.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.config import Config
    from portia.plan import Step
    from portia.tool import Tool
    from portia.workflow import Workflow


class ToollessChoice(BaseModel):
    """Represents a single choice option when multiple paths are available."""

    value: str = Field(description="The actual value/text of the choice")
    explanation: str = Field(description="Explanation of what this choice means")


class ToollessResponse(BaseModel):
    """Structured response from the toolless agent.

    This can handle various types of LLM tasks like:
    - Direct responses (summaries, analysis, recommendations)
    - Information extraction from context or past steps
    - Multiple choice situations requiring user input
    - Creative or generative content
    """

    response: str = Field(
        description="The main response text. This should be the complete answer or analysis.",
    )
    choices: list[ToollessChoice] | None = Field(
        default=None,
        description=(
            "Optional list of choices if the task requires user selection from multiple options"
        ),
    )


class ToolLessModel:
    """Model to call the toolless agent."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a powerful assistant specialized in various cognitive tasks, including:\n"
                    "- Summarizing information\n"
                    "- Extracting key details from context\n"
                    "- Providing analysis and recommendations\n"
                    "- Generating creative content\n"
                    "- Making decisions when given instruction to do so\n\n"
                    "When the task requires selection from multiple options, prefer to let the user choose:\n"
                    "1. Set requires_clarification to true\n"
                    "2. Include the options in the choices field\n"
                    "3. Provide clear explanations for each choice\n\n"
                    "For all other tasks:\n"
                    "1. Provide a complete, well-reasoned response\n"
                    "2. Use context and past steps when relevant\n"
                    "3. Stay focused on the specific task goal\n\n"
                    "A task requires selection if:\n"
                    "1. The previous step output is a list of options\n"
                    "2. The task is to select one of them\n"
                    "3. The task uses words like 'select', 'choose', 'pick', etc.\n\n"
                ),
            ),
            HumanMessagePromptTemplate.from_template(
                "Context from previous steps and current state:\n{context}\n\n"
                "Task to execute: {input}\n\n",
            ),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: BaseAgent) -> None:
        """Init the agent."""
        self.llm = llm.with_structured_output(ToollessResponse)
        self.context = context
        self.agent = agent

    def invoke(self, _: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        response = self.llm.invoke(
            self.prompt.format_messages(
                context=self.context,
                input=self.agent.step.task,
            ),
        )
        response = ToollessResponse.model_validate(response)
        return {"messages": [response.model_dump_json(indent=2)]}

    def _get_clarification_response(self) -> str:
        """Get any resolved clarification response."""
        if self.agent.workflow.outputs.clarifications:
            last_clarification = self.agent.workflow.outputs.clarifications[-1]
            if (
                last_clarification.resolved
                and last_clarification.step == self.agent.workflow.current_step_index
            ):
                return f"User selected: {last_clarification.response}"
        return "No previous clarification"


class ToolLessAgent(BaseAgent):
    """Agent for executing LLM-based tasks within an agentic workflow.

    This agent handles tasks that can be completed using only LLM capabilities, such as:
    - Summarizing information from previous steps
    - Extracting specific details from context
    - Providing analysis or recommendations
    - Generating creative content
    - Making decisions based on clear criteria

    When multiple valid approaches exist, it can request user clarification through choices.
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
        self.new_clarifications: list[Clarification] = []

    def clarifications_or_continue(
        self,
        state: MessagesState,
    ) -> Literal[AgentNode.TOOLLESS_AGENT, END]:  # type: ignore  # noqa: PGH003
        """Determine if clarification is needed or if we can proceed."""
        messages = state["messages"]
        last_message = messages[-1]
        response = ToollessResponse.model_validate_json(str(last_message.content))
        if response.choices:
            self.new_clarifications.append(
                MultipleChoiceClarification(
                    user_guidance="Please select one of the following options:",
                    argument_name="toolless_choice",
                    options=[choice.value for choice in response.choices],
                    step=self.workflow.current_step_index,
                ),
            )
            return END

        # If we have a valid response, end the workflow
        if response.response:
            return END

        # Only continue if we somehow don't have either
        return AgentNode.TOOLLESS_AGENT

    def execute_sync(self) -> Output:
        """Run the core execution logic of the task."""
        # Handle resolved clarifications
        if self.workflow.outputs.clarifications:
            last_clarification = self.workflow.outputs.clarifications[-1]
            if (
                isinstance(last_clarification, MultipleChoiceClarification)
                and last_clarification.resolved
                and last_clarification.step == self.workflow.current_step_index
            ):
                return Output(
                    value=f"Selected option: '{last_clarification.response}'",
                )

        context = self.get_system_context()
        llm = LLMWrapper(self.config).to_langchain()

        workflow = StateGraph(MessagesState)
        workflow.add_node(AgentNode.TOOLLESS_AGENT, ToolLessModel(llm, context, self).invoke)
        workflow.add_edge(START, AgentNode.TOOLLESS_AGENT)
        workflow.add_conditional_edges(
            AgentNode.TOOLLESS_AGENT,
            self.clarifications_or_continue,
        )

        app = workflow.compile()
        invocation_result = app.invoke({"messages": []})

        response = ToollessResponse.model_validate_json(invocation_result["messages"][-1].content)

        if self.new_clarifications:
            return Output(
                value=self.new_clarifications,
            )

        return Output(
            value=response.response,
        )
