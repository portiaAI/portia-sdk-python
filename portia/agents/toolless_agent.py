"""Agent designed for tasks that do not require external tools.

This agent is useful for solving tasks where the language model (LLM) intrinsically
has the necessary knowledge or for creative tasks. Any task that an LLM can handle
on its own, without the need for additional tools, can use the ToolLessAgent.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langgraph.graph import END, START, MessagesState, StateGraph

from portia.agents.base_agent import BaseAgent, Output
from portia.llm_wrapper import LLMWrapper


class ToolLessModel:
    """Model to invoke the toolless agent.

    This model uses the language model (LLM) to generate responses based on a
    predefined prompt, combining a system message and a user input message.
    It is invoked by the ToolLessAgent to perform tasks.

    Args:
        llm (BaseChatModel): The language model to use for generating responses.
        context (str): The context to be used when generating the response.
        agent (BaseAgent): The agent that manages the task.

    Methods:
        invoke(MessagesState): Invokes the LLM to generate a response based on the
                                current task and context.

    """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are very powerful assistant, but don't know current events."
                    " Answer the question from the user with the provided context."
                    " Keep your answer concise and to the point."
                ),
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ],
    )

    def __init__(self, llm: BaseChatModel, context: str, agent: BaseAgent) -> None:
        """Initialize the ToolLessModel.

        Args:
            llm (BaseChatModel): The language model to use for generating responses.
            context (str): The context to be used when generating the response.
            agent (BaseAgent): The agent that manages the task.

        """
        self.llm = llm
        self.context = context
        self.agent = agent

    def invoke(self, _: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        This method formats the input to the LLM using the current task and context,
        then generates a response.

        Args:
            _ (MessagesState): The message state (not used in this method).

        Returns:
            dict[str, Any]: A dictionary containing the model's response.

        """
        model = self.llm
        response = model.invoke(
            self.prompt.format_messages(
                input=self.agent.step.task + self.context,
            ),
        )

        return {"messages": [response]}


class ToolLessAgent(BaseAgent):
    """Agent responsible for achieving a task by using langgraph.

    This agent is designed to solve tasks that do not require any external tools.
    It leverages the ToolLessModel to generate responses based on the given task
    and context.

    Methods:
        execute_sync_with_context(context: str): Executes the core logic of the agent's task
        by invoking the ToolLessModel with the provided context.

        execute_sync(): Executes the core logic of the agent's task by invoking the
                        ToolLessModel with the appropriate inputs.

    """

    def execute_sync_with_context(self, context: str) -> Output:
        """Run the core execution logic of the task.

        This method generates a task-specific prompt and invokes the ToolLessModel
        within a StateGraph to produce a response based on the current context and task.

        Args:
            context (str): The context to be used when generating the response.

        Returns:
            Output: The result of the agent's execution, containing the generated response.

        """
        return self._execute_sync(context)


    def execute_sync(self) -> Output:
        """Run the core execution logic of the task.

        This method generates a task-specific prompt and invokes the ToolLessModel
        within a StateGraph to produce a response based on the current context and task.

        Returns:
            Output: The result of the agent's execution, containing the generated response.

        """
        return self._execute_sync(self.get_system_context())


    def _execute_sync(self, context: str) -> Output:
        """Run the core execution logic of the task with the provided context."""
        llm = LLMWrapper(self.config).to_langchain()
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
                    input=self.step.task,
                    clarification_prompt="",
                ),
            },
        )

        return Output(value=invocation_result["messages"][-1].content)
