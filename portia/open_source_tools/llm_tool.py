"""Tool for responding to prompts and completing tasks that don't require other tools."""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from portia.errors import ToolHardError
from portia.tool import Tool

if TYPE_CHECKING:
    from portia.execution_context import ExecutionContext

load_dotenv(override=True)


class LLMModel(Enum):
    """Enum for supported LLM models. Non exhaustive list."""

    # OpenAI Models
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"

    # Anthropic (Claude) Models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_2 = "claude-2"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_1 = "claude-1"

    # Mistral Models
    MISTRAL_TINY = "mistral-tiny"
    MISTRAL_SMALL = "mistral-small"
    MISTRAL_MEDIUM = "mistral-medium"
    MISTRAL_7B = "mistral-7b"
    MIXTRAL_8X7B = "mixtral-8x7b"


class LLMToolSchema(BaseModel):
    """Input for UserSelectionTool."""

    query: str = Field(...,
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
        "Jack of all trades used to respond to a prompt by relying solely on LLM capabilities. "
        "YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM only. "
        "This includes using your general knowledge, your in-built reasoning "
        "and your code interpreter capabilities."
    )
    args_schema: type[BaseModel] = LLMToolSchema
    output_schema: tuple[str, str] = (
        "str",
        "The LLM's response to the user query.",
    )

    def run(self, _: ExecutionContext, query: str) -> str:
        """Run the LLMTool."""
        # Set your API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        mistralai_api_key = os.getenv("MISTRALAI_API_KEY")
        system_prompt = """
            You are a Jack of all trades used to respond to a prompt by relying solely on LLM.
            capabilities. YOU NEVER CALL OTHER TOOLS. You use your native capabilities as an LLM
             only. This includes using your general knowledge, your in-built reasoning and
             your code interpreter capabilities.
            """

        # Initialize the model (replace with any compatible LangChain model)
        if openai_api_key:
            llm = ChatOpenAI(model_name=LLMModel.GPT_4.value, openai_api_key=openai_api_key)
        elif anthropic_api_key:
            llm = ChatAnthropic(
                model_name=LLMModel.CLAUDE_3_OPUS.value,
                anthropic_api_key=anthropic_api_key,
            )
        elif mistralai_api_key == "mistralai":
            llm = ChatMistralAI(
                model=LLMModel.MIXTRAL_8X7B.value,
                mistralai_api_key=mistralai_api_key,
            )
        else:
            raise ToolHardError("No API key found for any supported LLM model.")
        # Define system and user messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        # Get a response
        response = llm.invoke(messages)

        # Print the response
        return(response.content)
