"""Builder for Portia plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

from portia.builder.output import default_step_name
from portia.builder.portia_plan import PortiaPlan
from portia.builder.step import Hook, LLMStep, SingleToolAgent, ToolCall


class PlanBuilder:
    """Builder for Portia plans."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self.plan = PortiaPlan(steps=[])

    def llm_step(
        self,
        *,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        name: str | None = None,
    ) -> PlanBuilder:
        """Add a step that directly queries the LLM tool.

        Args:
            task: The task to perform.
            inputs: The inputs to the task.
            output_schema: The schema of the output.
            name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            LLMStep(
                task=task,
                inputs=inputs,
                output_schema=output_schema,
                name=name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def tool_call(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        output_schema: type[BaseModel] | None = None,
        name: str | None = None,
    ) -> PlanBuilder:
        """Add a step that directly invokes a tool.

        Args:
            tool: The tool to invoke.
            args: The arguments to the tool.
            output_schema: The schema of the output.
            name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            ToolCall(
                tool=tool,
                args=args,
                output_schema=output_schema,
                name=name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def single_tool_agent(
        self,
        *,
        tool: str,
        task: str,
        inputs: list[Any] | None = None,
        output_schema: type[BaseModel] | None = None,
        name: str | None = None,
    ) -> PlanBuilder:
        """Add a step that uses the execution agent with a tool.

        Args:
            tool: The tool to use.
            task: The task to perform.
            inputs: The inputs to the task.
            output_schema: The schema of the output.
            name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            SingleToolAgent(
                tool=tool,
                task=task,
                inputs=inputs,
                output_schema=output_schema,
                name=name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def hook(
        self,
        hook: Callable[..., bool],
        args: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> PlanBuilder:
        """Add a hook step.

        Args:
            hook: The hook to add.
            args: The args to call the hook with.
            name: Optional name for the step. If not provided, will be auto-generated.

        """
        self.plan.steps.append(
            Hook(
                hook=hook,
                args=args,
                name=name or default_step_name(len(self.plan.steps)),
            )
        )
        return self

    def build(self) -> PortiaPlan:
        """Return the plan, ready to run."""
        return self.plan
