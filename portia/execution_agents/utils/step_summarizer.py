"""StepSummarizer implementation.

The StepSummarizer can be used by agents to summarize the output of a given tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import MessagesState  # noqa: TC002
from pydantic import Field

from portia.execution_agents.output import LocalDataValue, Output
from portia.logger import logger
from portia.model import GenerativeModel, Message
from portia.planning_agents.context import get_tool_descriptions_for_tools
from portia.token_check import exceeds_context_threshold

if TYPE_CHECKING:
    from portia.config import Config
    from portia.model import GenerativeModel
    from portia.plan import Step
    from portia.tool import Tool


class StepSummarizer:
    """Class to summarize the output of a tool using llm.

    This is used only on the tool output message.

    Attributes:
        summarizer_prompt (ChatPromptTemplate): The prompt template used to generate the summary.
        model (GenerativeModel): The language model used for summarization.
        summary_max_length (int): The maximum length of the summary.
        step (Step): The step that produced the output.

    """

    summarizer_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    """
You are a highly skilled summarizer. Your task is to create a textual summary of the provided
tool output, make sure to follow the guidelines provided:
- Focus on the key information and maintain accuracy.
- Don't produce an overly long summary if it doesn't make sense.
- Make sure you capture ALL important information including sources and references.
- Large outputs will not be included. DO NOT summarise them but say that it is a large output.
- You might have multiple tool executions separated by 'OUTPUT_SEPARATOR'   .
- DO NOT INCLUDE 'OUTPUT_SEPARATOR' IN YOUR SUMMARY."""
                ),
            ),
            HumanMessagePromptTemplate.from_template(
                """
Here is original task:

{task_description}

- Make sure to not exceed the max limit of {max_length} characters.
- Do not reject the summary, even if the output is large - just do you best to provide a summary.
- Here is the description of the tool that produced the output:

    {tool_description}

- Please summarize the following tool output:

{tool_output}
""",
            ),
        ],
    )

    def __init__(
        self,
        config: Config,
        model: GenerativeModel,
        tool: Tool,
        step: Step,
        summary_max_length: int = 500,
    ) -> None:
        """Initialize the model.

        Args:
            config (Config): The configuration for the run.
            model (GenerativeModel): The language model used for summarization.
            tool (Tool): The tool used for summarization.
            step (Step): The step that produced the output.
            summary_max_length (int): The maximum length of the summary. Default is 500 characters.

        """
        self.config = config
        self.model = model
        self.summary_max_length = summary_max_length
        self.tool = tool
        self.step = step

    def invoke(self, state: MessagesState) -> dict[str, Any]:
        """Invoke the model with the given message state.

        This method processes the last message in the state, checks if it's a tool message with an
        output, and if so, generates a summary of the tool's output. The summary is then added to
        the artifact of the last message.

        Args:
            state (MessagesState): The current state of the messages, which includes the output.

        Returns:
            dict[str, Any]: A dict containing the updated message state, including the summary.

        Raises:
            Exception: If an error occurs during the invocation of the summarizer model.

        """
        messages = state["messages"]
        last_message = messages[-1] if len(messages) > 0 else None
        if not isinstance(last_message, ToolMessage) or not isinstance(
            last_message.artifact,
            Output,
        ):
            return {"messages": [last_message]}

        logger().debug(f"Invoke SummarizerModel on the tool output of {last_message.name}.")
        tool_messages = {msg.tool_call_id: msg for msg in messages if isinstance(msg, ToolMessage)}
        last_ai_message_with_tool_calls = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage) and msg.tool_calls),
            None,
        )
        tool_outputs = []
        if last_ai_message_with_tool_calls:
            for tool_call in last_ai_message_with_tool_calls.tool_calls:
                if tool_call["id"] in tool_messages:
                    tool_output_message = tool_messages[tool_call["id"]]
                    output = (
                        f"ToolCallName: {tool_call['name']}\n"
                        f"ToolCallArgs: {tool_call['args']}\n"
                        f"ToolCallOutput: {tool_output_message.content}"
                    )
                    tool_outputs.append(output)

        tool_output = "\nOUTPUT_SEPARATOR\n".join(tool_outputs)

        if exceeds_context_threshold(tool_output, self.config.get_summarizer_model(), 0.9):
            tool_output = (
                f"This is a large value (full length: {len(str(tool_output))} characters) "
                "which is held in agent memory."
            )
        messages = [
            Message.from_langchain(m)
            for m in self.summarizer_prompt.format_messages(
                tool_output=tool_output,
                max_length=self.summary_max_length,
                tool_description=get_tool_descriptions_for_tools([self.tool]),
                task_description=self.step.task,
            )
        ]
        structured_output_schema = (
            self.step.structured_output_schema or self.tool.structured_output_schema
        )
        if (
            structured_output_schema
            and not isinstance(tool_output, structured_output_schema)
            and isinstance(last_message.artifact, LocalDataValue)
        ):

            class SummarizerOutput(structured_output_schema):
                so_summary: str = Field(description="A summary of the tool output.")

            try:
                result = self.model.get_structured_response(messages, SummarizerOutput)
                last_message.artifact.summary = result.so_summary  # type: ignore[attr-defined]
                coerced_output = structured_output_schema.model_validate(result.model_dump())
                last_message.artifact.value = coerced_output
            except Exception as e:  # noqa: BLE001 - we want to catch all exceptions
                logger().error("Error in SummarizerModel invoke (Skipping summaries): " + str(e))

            return {"messages": [last_message]}

        try:
            response: Message = self.model.get_response(
                messages=messages,
            )
            summary = response.content
            last_message.artifact.summary = summary  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 - we want to catch all exceptions
            logger().error("Error in SummarizerModel invoke (Skipping summaries): " + str(e))

        return {"messages": [last_message]}
