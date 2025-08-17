"""Lightweight planning and execution helpers for simple agent workflows."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Sequence

from pydantic import BaseModel

from portia.config import Config
from portia.end_user import EndUser
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_hooks import ExecutionHooks
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, Step as PlanStep
from portia.plan_run import PlanRun
from portia.storage import InMemoryStorage
from portia.tool import ToolRunContext
from portia.tool_registry import DefaultToolRegistry, ToolRegistry


class Step:
    """Interface for all lightweight steps."""

    def run(self, runtime: "PortiaLite", outputs: list[Any]) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class LLMStep(Step):
    def __init__(self, query: str, output_class: type[BaseModel] | None = None) -> None:
        self.query = query
        self.output_class = output_class

    def run(self, runtime: "PortiaLite", outputs: list[Any]) -> bool:
        query = runtime._template(self.query, outputs)
        llm_tool = LLMTool(structured_output_schema=self.output_class)
        ctx = runtime._basic_ctx(query)
        result = llm_tool.run(ctx, task=query)
        outputs.append(result)
        return True


class ToolStep(Step):
    def __init__(self, tool: str, inputs: dict[str, Any]) -> None:
        self.tool = tool
        self.inputs = inputs

    def run(self, runtime: "PortiaLite", outputs: list[Any]) -> bool:
        tool = runtime.tool_registry.get_tool(self.tool)
        inputs = runtime._template(self.inputs, outputs)
        ctx = runtime._basic_ctx()
        result = tool.run(ctx, **inputs)
        outputs.append(result)
        return True


class LLMWithToolStep(Step):
    def __init__(
        self,
        tool: str,
        query: str,
        output_class: type[BaseModel] | None = None,
    ) -> None:
        self.tool = tool
        self.query = query
        self.output_class = output_class

    def run(self, runtime: "PortiaLite", outputs: list[Any]) -> bool:
        tool = runtime.tool_registry.get_tool(self.tool)
        query = runtime._template(self.query, outputs)
        step_obj = PlanStep(
            task=query,
            tool_id=tool.id,
            output="$output_0",
            structured_output_schema=self.output_class,
        )
        plan_obj = Plan(
            plan_context=PlanContext(query=query, tool_ids=[tool.id]),
            steps=[step_obj],
        )
        plan_run = PlanRun(plan_id=plan_obj.id, end_user_id=runtime.end_user.external_id)
        agent = DefaultExecutionAgent(
            plan=plan_obj,
            plan_run=plan_run,
            config=runtime.config,
            agent_memory=runtime.storage,
            end_user=runtime.end_user,
            tool=tool,
            execution_hooks=runtime.execution_hooks,
        )
        output_obj = agent.execute_sync()
        outputs.append(output_obj.value)
        return True


class Hook(Step):
    def __init__(self, callable: Callable[[Sequence[Any]], bool]) -> None:
        self.callable = callable

    def run(self, runtime: "PortiaLite", outputs: list[Any]) -> bool:
        return self.callable(outputs)


class PlanBuilderLite:
    """Builder for lightweight step plans.

    Steps can be chained together and then executed with :class:`PortiaLite`.
    Only a subset of the full :class:`PlanBuilder` API is implemented.
    """

    def __init__(self) -> None:
        self._steps: list[Step] = []

    def llm_with_tool_step(
        self,
        *,
        tool: str,
        query: str,
        output_class: type[BaseModel] | None = None,
    ) -> "PlanBuilderLite":
        """Add a step that uses the execution agent with a tool."""

        self._steps.append(LLMWithToolStep(tool=tool, query=query, output_class=output_class))
        return self

    def tool_step(self, *, tool: str, inputs: dict[str, Any]) -> "PlanBuilderLite":
        """Add a step that directly invokes a tool."""

        self._steps.append(ToolStep(tool=tool, inputs=inputs))
        return self

    def llm_step(
        self,
        *,
        query: str,
        output_class: type[BaseModel] | None = None,
    ) -> "PlanBuilderLite":
        """Add a step that directly queries the LLM tool."""

        self._steps.append(LLMStep(query=query, output_class=output_class))
        return self

    def hook(self, callable: Callable[[Sequence[Any]], bool]) -> "PlanBuilderLite":
        """Add a hook step.

        The callable receives the list of previous outputs and should return ``True``
        to continue execution or ``False`` to stop execution.
        """

        self._steps.append(Hook(callable))
        return self

    def build(self) -> list[Step]:
        """Return the list of steps representing the plan."""

        return self._steps


class PortiaLite:
    """Minimal helper for running small plans built with :class:`PlanBuilderLite`."""

    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | list[Any] | None = None,
    ) -> None:
        self.config = config if config else Config.from_default()
        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, list):
            self.tool_registry = ToolRegistry(tools)
        else:
            self.tool_registry = DefaultToolRegistry(self.config)
        self.storage = InMemoryStorage()
        self.execution_hooks = ExecutionHooks()
        self.end_user = EndUser(external_id="portia:lite_user")

    def _template(self, value: Any, outputs: Sequence[Any]) -> Any:
        """Template ``$outputX`` occurrences in strings using previous outputs."""

        if isinstance(value, str):
            def repl(match: re.Match[str]) -> str:
                idx = int(match.group(1))
                return str(outputs[idx])

            return re.sub(r"\$output(\d+)", repl, value)
        if isinstance(value, dict):
            return {k: self._template(v, outputs) for k, v in value.items()}
        if isinstance(value, list):
            return [self._template(v, outputs) for v in value]
        return value

    def _basic_ctx(self, query: str = "") -> ToolRunContext:
        """Create a minimal :class:`ToolRunContext` for direct tool invocations."""

        step = PlanStep(task=query, output="$output_0")
        plan = Plan(plan_context=PlanContext(query=query, tool_ids=[]), steps=[step])
        plan_run = PlanRun(plan_id=plan.id, end_user_id=self.end_user.external_id)
        return ToolRunContext(
            end_user=self.end_user,
            plan_run=plan_run,
            plan=plan,
            config=self.config,
            clarifications=[],
        )

    def run(self, plan: Sequence[Step]) -> Any:
        """Execute the provided plan and return the final output."""

        outputs: list[Any] = []
        for step in plan:
            cont = step.run(self, outputs)
            if not cont:
                break
        return outputs[-1] if outputs else None
