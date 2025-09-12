"""Implementation of the LLM step."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from portia.builder.step_v2 import StepV2

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override  # pragma: no cover

from langsmith import traceable
from pydantic import BaseModel, Field

from portia.model import GenerativeModel  # noqa: TC001
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Step
from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from portia.builder.plan_v2 import PlanV2
    from portia.run_context import RunContext


class LLMStep(StepV2):
    """A step that executes a task using an LLM without any tool access.

    This step is used for pure language model tasks like text generation,
    analysis, or transformation that don't require external tool calls.
    """

    task: str = Field(description="The natural language task for the LLM to perform.")
    inputs: list[Any] = Field(
        default_factory=list,
        description=(
            "Additional context data for the task. Can include references to previous step "
            "outputs (using StepOutput) or plan inputs (using Input), or literal values. "
            "These are provided as context to help the LLM complete the task."
        ),
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Pydantic model class defining the expected structure of the LLM's response. "
            "If provided, the output from the LLM will be coerced to match this schema."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Custom system prompt to guide the LLM's behavior. If not specified, "
            "the default LLMTool system prompt will be used."
        ),
    )
    model: GenerativeModel | str | None = Field(
        default=None,
        description=(
            "The model to use for this step. If not provided, the default model from the config "
            "will be used."
        ),
    )

    def __str__(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_schema.__name__}" if self.output_schema else ""
        return f"LLMStep(task='{self.task}'{output_info})"

    @override
    @traceable(name="LLM Step - Run")
    async def run(self, run_data: RunContext) -> str | BaseModel:  # pyright: ignore[reportIncompatibleMethodOverride] - needed due to Langsmith decorator
        """Execute the LLM task and return its response."""
        if self.system_prompt:
            llm_tool = LLMTool(
                structured_output_schema=self.output_schema,
                prompt=self.system_prompt,
                model=self.model,
            )
        else:
            llm_tool = LLMTool(structured_output_schema=self.output_schema, model=self.model)
        wrapped_tool = ToolCallWrapper(
            child_tool=llm_tool,
            storage=run_data.storage,
            plan_run=run_data.plan_run,
        )

        tool_ctx = run_data.get_tool_run_ctx()
        task_data = self._resolve_input_references_with_descriptions(self.inputs, run_data)
        task = self._template_references(self.task, run_data)
        return await wrapped_tool.arun(tool_ctx, task=task, task_data=task_data)

    @override
    def to_legacy_step(self, plan: PlanV2) -> Step:
        """Convert this LLMStep to a legacy Step."""
        return Step(
            task=self.task,
            inputs=self._inputs_to_legacy_plan_variables(self.inputs, plan),
            tool_id=LLMTool.LLM_TOOL_ID,
            output=plan.step_output_name(self),
            structured_output_schema=self.output_schema,
            condition=self._get_legacy_condition(plan),
        )
