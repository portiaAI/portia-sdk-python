"""Tool for responding to prompts and completing tasks that don't require other tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

from portia.config import Config, LLMModel, LLMProvider
from portia.llm_wrapper import LLMWrapper
from portia.tool import Tool

if TYPE_CHECKING:
    from portia.execution_context import ExecutionContext


class LLMToolSchema(BaseModel):
    """Input for LLM Tool."""

    task: str = Field(
        ...,
        description="The task to be completed by the LLM tool.",
    )


class LLMTool(Tool[str]):
    """General purpose LLM tool. Customizable to user requirements. Won't call other tools."""

    id: str = "llm_tool"
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
    # Provide customizable parameters for the LLM so that it can be initialized
    # per user requirements
    model_name: str = LLMModel.GPT_4_O.value
    provider: str = LLMProvider.OPENAI.value
    temperature: float = 0.0
    seed: int = 42
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

    def run(self, ctx: ExecutionContext, task: str) -> str:
        """Run the LLMTool."""
        config = Config.from_default(
            model_name=self.model_name,
            provider=self.provider,
            temperature=self.temperature,
            seed=self.seed,
        )
        llm_wrapper = LLMWrapper(config)
        llm = llm_wrapper.to_langchain()
        # Define system and user messages
        context = (
            "Additional context for the LLM tool to use to complete the task, provided by the "
            "workflow run information and results of other tool calls. Use this to resolve any "
            "tasks"
        )
        if ctx.workflow_run_context:
            context += f"\nWorkflow run context: {ctx.workflow_run_context}"
        if self.tool_context:
            context += f"\nTool context: {self.tool_context}"
        content = task if not len(context.split("\n")) > 1 else f"{context}\n\n{task}"
        messages = [
            HumanMessage(content=self.prompt),
            HumanMessage(content=content),
        ]
        response = llm.invoke(messages)
        return str(response.content)

    @classmethod
    def from_config(cls, config: Config) -> LLMTool:
        """Create an LLMTool from a config."""
        return cls(
            model_name=config.llm_model_name.value,
            provider=config.llm_provider.value,
            temperature=config.llm_model_temperature,
            seed=config.llm_model_seed,
        )
