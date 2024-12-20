"""Agents for doing things."""

from __future__ import annotations

from abc import abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from portia.agents.context import build_context
from portia.clarification import Clarification, InputClarification

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel

    from portia.plan import Output, Variable
    from portia.tool import Tool


class RequestClarificationInput(BaseModel):
    """Input arguments for RequestClarification Tool."""

    missing_args: list[str] = Field(
        description="Arguments required by other tools without values in the context or input.",
    )


class RequestClarificationTool(BaseTool):
    """RequestClarification Tool."""

    name: str = "RequestClarification"
    description: str = (
        "Use this tool when in doubt about an argument or parameter of a different tool."
    )
    args_schema: type[BaseModel] = RequestClarificationInput
    return_direct: bool = True

    def _run(
        self,
        missing_args: list[str],
        _: CallbackManagerForToolRun | None = None,
    ) -> list[Clarification]:
        """Use the tool."""
        return [
            InputClarification(
                argument_name=arg,
                user_guidance=f"Missing argument: {arg}",
            )
            for arg in missing_args
        ]

    async def _arun(
        self,
        missing_args: list[str],
        _: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool asynchronously."""
        error = f"Request clarification does not support async, requested: {missing_args}"
        raise NotImplementedError(error)


class BaseAgent:
    """Base agent that can be implemented by different mechanisms."""

    def __init__(
        self,
        description: str,
        inputs: list[Variable],
        clarifications: list[Clarification] | None = None,
        tool: Tool | None = None,
        system_context: list[str] | None = None,
    ) -> None:
        """Initialize the base agent."""
        self.description = description
        self.inputs = inputs
        self.tool = tool
        self.clarifications = clarifications
        if system_context is None:
            self.system_context = self._default_system_context()
        else:
            self.system_context = system_context

    @abstractmethod
    def execute_sync(self, llm: BaseChatModel, step_outputs: dict[str, Output]) -> Output:
        """Run the core execution logic of the task."""

    def _get_context(self, step_outputs: dict[str, Output]) -> str:
        """Turn inputs and past outputs into a context string for the agent."""
        return build_context(self.inputs, step_outputs, self.clarifications, self.system_context)

    def _default_system_context(self) -> list[str]:
        """Provide default system context."""
        today = f"Today's date is {datetime.now(UTC).strftime('%Y-%m-%d')}"
        return [today]
