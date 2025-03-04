"""Tool for responding to prompts and completing tasks that don't require other tools."""

from __future__ import annotations

from typing import ClassVar

from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

from portia.llm_wrapper import LLMWrapper
from portia.tool import Tool, ToolRunContext


class LLMToolSchema(BaseModel):
    """Input for LLM Tool."""

    task: str = Field(
        ...,
        description="The task to be completed by the LLM tool.",
    )


class LLMTool(Tool[str]):
    """General purpose LLM tool. Customizable to user requirements. Won't call other tools."""

    LLM_TOOL_ID: ClassVar[str] = "llm_tool"
    id: str = LLM_TOOL_ID
    name: str = "LLM Tool"
    description: str = (
        "Jack of all trades tool to respond to a prompt by relying solely on LLM capabilities. "
        "YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM only. "
        "This includes using your general knowledge, your in-built reasoning "
        "and your code interpreter capabilities. This tool can be used to summarize the outputs of "
        "other tools, make general language model queries or to answer questions. This should be "
        "used only as a last resort when no other tool satisfies a step in a task, however if "
        "there are no other tools that can be used to complete a step or for steps that don't "
        "require a tool call, this SHOULD be used"
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

    def run(self, ctx: ToolRunContext, task: str) -> str:
        """Run the LLMTool."""
        llm_wrapper = LLMWrapper(ctx.config)
        llm = llm_wrapper.to_langchain()

        # Define system and user messages
        context = (
            "Additional context for the LLM tool to use to complete the task, provided by the "
            "run information and results of other tool calls. Use this to resolve any "
            "tasks"
        )
        if ctx.execution_context.plan_run_context:
            context += f"\nRun context: {ctx.execution_context.plan_run_context}"
        if self.tool_context:
            context += f"\nTool context: {self.tool_context}"
        content = task if not len(context.split("\n")) > 1 else f"{context}\n\n{task}"
        messages = [
            HumanMessage(content=self.prompt),
            HumanMessage(content=content),
        ]
        response = llm.invoke(messages)
        return str(response.content)
