"""Agent designed when no tool is needed."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph

from portia.agents.base_agent import BaseAgent
from portia.plan import Output


class ToolLessModel:
    """Model to call the toolless agent."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are very powerful assistant, but don't know current events."
                    "Answer the question from the user with the provided context."
                    "Keep your answer concise and to the point."
                ),
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: BaseAgent) -> None:
        """Init the agent."""
        self.llm = llm
        self.context = context
        self.agent = agent

    def invoke(self, _: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state."""
        model = self.llm
        response = model.invoke(
            self.prompt.format_messages(
                input=self.agent.description + self.context,
            ),
        )

        return {"messages": [response]}


class ToolLessAgent(BaseAgent):
    """Agent responsible for achieving a task by using langgraph."""

    def execute_sync(self, llm: BaseChatModel, step_outputs: dict[str, Output]) -> Output:
        """Run the core execution logic of the task."""
        context = self._get_context(step_outputs)

        task_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are very powerful assistant, but don't know current events. "
                        "{clarification_prompt}"
                    ),
                ),
                ("system", "{context}"),
                ("human", "{input}"),
            ],
        )

        workflow = StateGraph(MessagesState)

        # The agent node is the only node in the graph
        workflow.add_node("agent", ToolLessModel(llm, context, self).invoke)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        app = workflow.compile()
        invocation_result = app.invoke(
            {
                "messages": task_prompt.format_messages(
                    context=context,
                    input=self.description,
                    clarification_prompt="",
                ),
            },
        )

        return Output(value=invocation_result["messages"][-1].content)
