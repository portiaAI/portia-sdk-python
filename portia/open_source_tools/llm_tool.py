"""Tool for responding to prompts and completing tasks that don't require other tools."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from portia.logger import logger
from portia.model import GenerativeModel, Message
from portia.telemetry.telemetry_service import ProductTelemetry
from portia.telemetry.views import LLMToolUsageTelemetryEvent
from portia.tool import Tool, ToolRunContext


class LLMToolSchema(BaseModel):
    """Input for LLM Tool."""

    task: str = Field(
        ...,
        description="The task to be completed by the LLM tool",
    )
    task_data: dict[str, Any] | list[Any] | str | None = Field(
        default=None,
        description="Task data that should be used to complete the task. "
        "Can be a string, a list of strings, a dictionary of strings or "
        "a list of objects that will be converted to strings. "
        "Important: This should include all relevant data in their entirety, "
        "from the first to the last character (i.e. NOT a summary).",
    )


class LLMTool(Tool[str | BaseModel]):
    """General purpose LLM tool. Customizable to user requirements. Won't call other tools."""

    LLM_TOOL_ID: ClassVar[str] = "llm_tool"
    id: str = LLM_TOOL_ID
    name: str = "LLM Tool"
    description: str = (
        "Jack of all trades tool to respond to a prompt by relying solely on LLM capabilities. "
        "YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM only. "
        "This includes using your general knowledge and your in-built reasoning. "
        "This tool can be used to summarize the outputs of "
        "other tools, make general language model queries or to answer questions. This should be "
        "used only as a last resort when no other tool satisfies a step in a task, however if "
        "there are no other tools that can be used to complete a step or for steps that don't "
        "require a tool call, this SHOULD be used. "
        "MAKE SURE the task_data includes ALL INPUT VARIABLES IN THE CONTEXT. "
        "DO NOT use this tool if you require input from user."
    )
    args_schema: type[BaseModel] = LLMToolSchema
    output_schema: tuple[str, str] = (
        "str",
        "The LLM's response to the user query.",
    )
    prompt: str = """
        You are a Jack of all trades used to respond to a prompt by relying solely on LLM.
        capabilities. YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM
         only. This includes using your general knowledge, your in-built reasoning and
         your code interpreter capabilities. You exist as part of a wider system of tool calls
         for a multi-step task to be used to answers questions, summarize outputs of other tools
         and to make general language model queries. You might not have all the context of the
         wider task, so you should use your general knowledge and reasoning capabilities to make
         educated guesses and assumptions where you don't have all the information. Be concise and
         to the point.
        """
    tool_context: str = ""

    model: GenerativeModel | str | None = Field(
        default=None,
        exclude=True,
        description="The model to use for the LLMTool. If not provided, "
        "the model will be resolved from the config.",
    )
    structured_output_schema: type[BaseModel] | None = Field(
        default=None,
        description="The schema to use for the structured output of the LLMTool. "
        "If not provided, the output will be a string.",
    )

    @staticmethod
    def process_task_data(task_data: dict[str, Any] | list[Any] | str | None) -> str:
        """Process task_data into a string, handling different input types.

        Args:
            task_data: Data that can be a None, a string, a list of objects or
            a dictionary of objects.

        Returns:
            A string representation of the data, with list items joined by newlines.

        """
        if task_data is None:
            return ""

        if isinstance(task_data, str):
            return task_data

        if isinstance(task_data, dict):
            return "\n".join(f"{key}: {value}" for key, value in task_data.items())

        return "\n".join(str(item) for item in task_data)

    def run(
        self,
        ctx: ToolRunContext,
        task: str,
        task_data: dict[str, Any] | list[Any] | str | None = None,
    ) -> str | BaseModel:
        """Run the LLMTool."""
        model = ctx.config.get_generative_model(self.model) or ctx.config.get_default_model()

        telemetry = ProductTelemetry()
        telemetry.capture(LLMToolUsageTelemetryEvent(model=str(model), sync=True))

        messages = self._get_messages(task, task_data)
        logger().trace("LLM call: llm-tool")
        if self.structured_output_schema:
            return model.get_structured_response(messages, self.structured_output_schema)

        response = model.get_response(messages)
        return str(response.content)

    async def arun(
        self,
        ctx: ToolRunContext,
        task: str,
        task_data: dict[str, Any] | list[Any] | str | None = None,
    ) -> str | BaseModel:
        """Run the LLMTool asynchronously."""
        model = ctx.config.get_generative_model(self.model) or ctx.config.get_default_model()

        telemetry = ProductTelemetry()
        telemetry.capture(LLMToolUsageTelemetryEvent(model=str(model), sync=False))

        messages = self._get_messages(task, task_data)
        logger().trace("LLM call: llm-tool")
        if self.structured_output_schema:
            return await model.aget_structured_response(messages, self.structured_output_schema)
        response = await model.aget_response(messages)
        return str(response.content)

    def _get_messages(
        self,
        task: str,
        task_data: dict[str, Any] | list[Any] | str | None = None,
    ) -> list[Message]:
        """Get the messages for the LLMTool."""
        context = (
            "Additional context for the LLM tool to use to complete the task, provided by the "
            "run information and results of other tool calls. Use this to resolve any "
            "tasks"
        )

        task_data_str = self.process_task_data(task_data)

        task_str = task
        if task_data_str:
            task_str += f"\nTask data: {task_data_str}"
        if self.tool_context:
            context += f"\nTool context: {self.tool_context}"
        content = task_str if not len(context.split("\n")) > 1 else f"{context}\n\n{task_str}"
        return [
            Message(role="user", content=self.prompt),
            Message(role="user", content=content),
        ]
