"""Tool for responding to prompts and completing tasks that don't require other tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from portia.config import Config, LLMModel, LLMProvider
from portia.llm_wrapper import LLMWrapper
from portia.tool import Tool

if TYPE_CHECKING:
    from portia.execution_context import ExecutionContext

load_dotenv(override=True)


class LLMToolSchema(BaseModel):
    """Input for UserSelectionTool."""

    query: str = Field(
        ...,
        description="The user query",
    )


class LLMTool(Tool[str]):
    """Jack of all trades used to respond to a prompt by relying solely on LLM capabilities.

    YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM only.
    This includes using your general knowledge, your in-built reasoning and
     your code interpreter capabilities.
    """

    id: str = "llm_tool"
    name: str = "LLM Tool"
    description: str = (
        "Jack of all trades tool to respond to a prompt by relying solely on LLM capabilities. "
        "YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM only. "
        "This includes using your general knowledge, your in-built reasoning "
        "and your code interpreter capabilities."
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
         your code interpreter capabilities.
        """

    def run(self, _: ExecutionContext, query: str) -> str:
        """Run the LLMTool."""
        # Initialize the model (replace with any compatible LangChain model)
        config = Config.from_default(
            model_name=self.model_name,
            provider=self.provider,
            temperature=self.temperature,
            seed=self.seed,
        )
        llm_wrapper = LLMWrapper(config)
        llm = llm_wrapper.to_langchain()
        # Define system and user messages
        messages = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=query),
        ]

        # Get a response
        response = llm.invoke(messages)

        # Print the response
        return str(response.content)
