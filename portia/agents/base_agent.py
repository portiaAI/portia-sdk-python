"""Agents are responsible for executing steps of a workflow.

The BaseAgent class is the base class that all agents must extend.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date, datetime
from enum import Enum
import json
from typing import TYPE_CHECKING, Generic

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from portia.agents.context import build_context
from portia.common import SERIALIZABLE_TYPE_VAR
from portia.execution_context import get_execution_context

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Step
    from portia.tool import Tool
    from portia.workflow import Workflow


class BaseAgent:
    """An Agent is responsible for carrying out the task defined in the given Step.

    This Base agent is the class all agents must extend. Critically, agents must implement the
    execute_sync function which is responsible for actually carrying out the task as given in
    the step. They have access to copies of the step, workflow, and config but changes to those
    objects are forbidden.

    Optionally, new agents may also override the get_context function, which is responsible for
    the system context for the agent. This should be done with thought, as the details of the system
    context are critically important for LLM performance.
    """

    def __init__(
        self,
        step: Step,
        workflow: Workflow,
        config: Config,
        tool: Tool | None = None,
    ) -> None:
        """Initialize the base agent with the given args.

        Importantly, the models here are frozen copies of those used in the Runner.
        They are meant as read-only references, useful for execution of the task
        but cannot be edited. The agent should return output via the response
        of the execute_sync method.

        Args:
            step (Step): The step that defines the task to be executed.
            workflow (Workflow): The workflow that contains the step and related data.
            config (Config): The configuration settings for the agent.
            tool (Tool | None): An optional tool associated with the agent (default is None).

        """
        self.step = step
        self.tool = tool
        self.config = config
        self.workflow = workflow

    @abstractmethod
    def execute_sync(self) -> Output:
        """Run the core execution logic of the task synchronously.

        Implementation of this function is deferred to individual agent implementations,
        making it simple to write new ones.

        Returns:
            Output: The output of the task execution.

        """

    def get_system_context(self) -> str:
        """Build a generic system context string from the step and workflow provided.

        This function retrieves the execution context and generates a system context
        based on the step and workflow provided to the agent.

        Returns:
            str: A string containing the system context for the agent.

        """
        ctx = get_execution_context()
        return build_context(
            ctx,
            self.step,
            self.workflow,
        )


class Output(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Output of a tool with a wrapper for data, summaries, and LLM interpretation.

    This class contains a generic value `T` bound to `Serializable`.

    Attributes:
        value (SERIALIZABLE_TYPE_VAR | None): The output of the tool.
        summary (str | None): A textual summary of the output. Not all tools generate summaries.

    """

    model_config = ConfigDict(extra="forbid")

    value: SERIALIZABLE_TYPE_VAR | None = Field(default=None, description="The output of the tool")
    summary: str | None = Field(
        default=None,
        description="Textual summary of the output of the tool. Not all tools generate summaries.",
    )

    @field_serializer("value")
    def serialize_value(self, value: SERIALIZABLE_TYPE_VAR | None) -> str:  # noqa: PLR0911
        """Serialize the value to a string.

        Args:
            value (SERIALIZABLE_TYPE_VAR | None): The value to serialize.

        Returns:
            str: The serialized value as a string.

        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)  # Ensure proper JSON formatting

        if isinstance(value, set):
            return json.dumps(
                list(value),
                ensure_ascii=False,
            )  # Convert set to list before serialization

        if isinstance(value, (int, float, bool)):
            return json.dumps(value, ensure_ascii=False)  # Ensures booleans become "true"/"false"

        if isinstance(value, (datetime, date)):
            return value.isoformat()  # Convert date/time to ISO format

        if isinstance(value, Enum):
            return str(value.value)  # Convert Enums to their values

        if isinstance(value, BaseModel):
            return value.model_dump_json()  # Use Pydantic's built-in serialization for models

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")  # Convert bytes to string

        return str(value)  # Fallback for other types
