"""Agents are responsible for executing steps of a workflow.

The BaseAgent class is the base class all agents must extend.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic

from pydantic import BaseModel, Field

from portia.agents.context import build_context
from portia.types import SERIALIZABLE_TYPE_VAR

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Step
    from portia.tool import Tool
    from portia.workflow import Workflow


class BaseAgent:
    """An Agent is responsible for carrying out the task defined in the given Step.

    This Base agent is the class all agents must extend. Critically agents must implement the
    execute_sync function which is responsible for actually carrying out the task as given in
    the step. They have access to copies of the step, workflow and config but changes to those
    objects are forbidden.

    Optionally agents may also override the get_context function which is responsible for setting
    the system_context for the agent.
    """

    def __init__(
        self,
        step: Step,
        workflow: Workflow,
        config: Config,
        tool: Tool | None = None,
    ) -> None:
        """Initialize the base agent with the given args.

        Importantly the models here are frozen copies of those used in the Runner.
        They are meant as a read only reference, useful for execution of the task
        but can not be edited. The agent should return output via the response
        of the execute_sync method.
        """
        self.step = step
        self.tool = tool
        self.config = config
        self.workflow = workflow

    @abstractmethod
    def execute_sync(self) -> Output:
        """Run the core execution logic of the task synchronously.

        Implementation of this function is deferred to individual agent implementations
        making it simple to write new ones.
        """

    def get_system_context(self) -> str:
        """Build a generic system context string from the step and workflow provided."""
        return build_context(
            self.step.inputs,
            self.workflow.step_outputs,
            self.workflow.clarifications,
            self.config.agent_system_context_extension,
        )


class Output(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Output of a tool with wrapper for data, summaries and LLM interpretation.

    Contains a generic value T bound to Serializable.
    """

    value: SERIALIZABLE_TYPE_VAR | None = Field(default=None, description="The output of the tool")
