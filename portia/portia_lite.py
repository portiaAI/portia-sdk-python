"""Lightweight planning and execution helpers for simple agent workflows."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from portia.tool_wrapper import ToolCallWrapper

if TYPE_CHECKING:
    from pydantic import BaseModel

from collections.abc import Callable, Sequence

from langsmith import traceable

from portia.config import Config, GenerativeModelsConfig, StorageClass
from portia.end_user import EndUser
from portia.execution_agents.default_execution_agent import DefaultExecutionAgent
from portia.execution_agents.output import LocalDataValue
from portia.introspection_agents.default_introspection_agent import DefaultIntrospectionAgent
from portia.introspection_agents.introspection_agent import (
    PreStepIntrospectionOutcome,
)
from portia.logger import logger, logger_manager
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Plan, PlanContext, ReadOnlyPlan
from portia.plan import Step as PlanStep
from portia.plan_run import PlanRun, PlanRunState, ReadOnlyPlanRun
from portia.storage import DiskFileStorage, InMemoryStorage, PortiaCloudStorage
from portia.tool import ToolRunContext
from portia.tool_registry import DefaultToolRegistry, ToolRegistry
from portia.version import get_version


class Step:
    """Interface for all lightweight steps."""

    def __init__(self, condition: str | Callable[[list[Any]], bool] | None = None) -> None:
        """Initialize the step with an optional condition."""
        self.condition = condition

    def run(self, runtime: PortiaLite, outputs: list[Any]) -> bool:  # pragma: no cover - interface
        """Execute the step and return whether to continue execution."""
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - interface
        """Return a description of this step for logging purposes."""
        raise NotImplementedError

    def to_portia_step(self, step_index: int) -> PlanStep | None:  # pragma: no cover - interface
        """Convert this step to a PlanStep from plan.py.

        Args:
            step_index: The index of this step in the plan (used for generating output names)

        Returns:
            PlanStep: A converted PlanStep object

        """
        raise NotImplementedError

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

    def _basic_ctx(self, runtime: PortiaLite) -> ToolRunContext:
        """Create a minimal :class:`ToolRunContext` for direct tool invocations."""
        return ToolRunContext(
            end_user=runtime.end_user,
            plan_run=runtime.current_plan_run,
            plan=runtime.current_plan,
            config=runtime.config,
            clarifications=[],
        )


class LLMStep(Step):
    """A step that runs a given query through an LLM."""

    def __init__(
        self,
        query: str,
        output_class: type[BaseModel] | None = None,
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> None:
        """Initialize the LLM step."""
        super().__init__(condition)
        self.query = query
        self.output_class = output_class

    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_class.__name__}" if self.output_class else ""
        return f"LLMStep(query='{self.query}'{output_info})"

    @traceable(name="LLM Step Run")
    def run(self, runtime: PortiaLite, outputs: list[Any]) -> bool:
        """Run the LLM query."""
        query = self._template(self.query, outputs)
        llm_tool = LLMTool(structured_output_schema=self.output_class)
        ctx = self._basic_ctx(runtime)
        result = llm_tool.run(ctx, task=query)
        outputs.append(result)
        return True

    def to_portia_step(self, step_index: int) -> PlanStep:
        """Convert this LLMStep to a PlanStep."""
        # Only use condition if it's a string, otherwise set to None
        condition = self.condition if isinstance(self.condition, str) else None
        return PlanStep(
            task=self.query,
            inputs=[],
            tool_id=LLMTool.LLM_TOOL_ID,
            output=f"$output_{step_index}",
            condition=condition,
            structured_output_schema=self.output_class,
        )


class ToolStep(Step):
    """A step that runs a tool with the given inputs."""

    def __init__(
        self,
        tool: str,
        inputs: dict[str, Any],
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> None:
        """Initialize the tool step."""
        super().__init__(condition)
        self.tool = tool
        self.inputs = inputs

    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        return f"ToolStep(tool='{self.tool}', inputs={self.inputs})"

    @traceable(name="Tool Step Run")
    def run(self, runtime: PortiaLite, outputs: list[Any]) -> bool:
        """Run the tool."""
        tool = runtime.tool_registry.get_tool(self.tool)
        wrapped_tool = ToolCallWrapper(
            child_tool=tool,
            storage=runtime.storage,
            plan_run=runtime.current_plan_run,
        )
        inputs = self._template(self.inputs, outputs)
        ctx = self._basic_ctx(runtime)
        result = wrapped_tool.run(ctx, **inputs)
        outputs.append(result)
        return True

    def to_portia_step(self, step_index: int) -> PlanStep:
        """Convert this ToolStep to a PlanStep."""
        # Create a task description from the tool and inputs
        inputs_desc = ", ".join([f"{k}={v}" for k, v in self.inputs.items()])
        task = f"Use tool {self.tool} with inputs: {inputs_desc}"

        # Only use condition if it's a string, otherwise set to None
        condition = self.condition if isinstance(self.condition, str) else None
        return PlanStep(
            task=task,
            inputs=[],
            tool_id=self.tool,
            output=f"$output_{step_index}",
            condition=condition,
            structured_output_schema=None,
        )


class AgentStep(Step):
    """A step that runs a tool with a query through the execution agent."""

    def __init__(
        self,
        tool: str,
        query: str,
        output_class: type[BaseModel] | None = None,
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> None:
        """Initialize the agent step."""
        super().__init__(condition)
        self.tool = tool
        self.query = query
        self.output_class = output_class

    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        output_info = f" -> {self.output_class.__name__}" if self.output_class else ""
        return f"AgentStep(tool='{self.tool}', query='{self.query}'{output_info})"

    @traceable(name="Agent Step Run")
    def run(self, runtime: PortiaLite, outputs: list[Any]) -> bool:
        """Run the agent step."""
        tool = runtime.tool_registry.get_tool(self.tool)
        wrapped_tool = ToolCallWrapper(
            child_tool=tool,
            storage=runtime.storage,
            plan_run=runtime.current_plan_run,
        )
        agent = DefaultExecutionAgent(
            plan=runtime.current_plan,
            plan_run=runtime.current_plan_run,
            config=runtime.config,
            agent_memory=runtime.storage,
            end_user=runtime.end_user,
            tool=wrapped_tool,
        )
        output_obj = agent.execute_sync()
        outputs.append(output_obj.get_value())
        return True

    def to_portia_step(self, step_index: int) -> PlanStep:
        """Convert this AgentStep to a PlanStep."""
        # Only use condition if it's a string, otherwise set to None
        condition = self.condition if isinstance(self.condition, str) else None
        return PlanStep(
            task=self.query,
            inputs=[],
            tool_id=self.tool,
            output=f"$output_{step_index}",
            condition=condition,
            structured_output_schema=self.output_class,
        )


class Hook(Step):
    """A step that runs a hook."""

    def __init__(
        self,
        hook: Callable[..., bool],
        inputs: dict[str, Any] | None = None,
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> None:
        """Initialize the hook."""
        super().__init__(condition)
        self.hook = hook
        self.inputs = inputs or {}

    def describe(self) -> str:
        """Return a description of this step for logging purposes."""
        hook_name = getattr(self.hook, "__name__", str(self.hook))
        if self.inputs:
            return f"Hook(hook={hook_name}, inputs={self.inputs})"
        return f"Hook(hook={hook_name})"

    @traceable(name="Hook Run")
    def run(self, runtime: PortiaLite, outputs: list[Any]) -> bool:  # noqa: ARG002
        """Run the hook."""
        templated_inputs = self._template(self.inputs, outputs)
        return self.hook(**templated_inputs)

    def to_portia_step(self, step_index: int) -> PlanStep:
        """Convert this Hook to a PlanStep."""
        return None


class PlanBuilderLite:
    """Builder for lightweight step plans.

    Steps can be chained together and then executed with :class:`PortiaLite`.
    Only a subset of the full :class:`PlanBuilder` API is implemented.
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._steps: list[Step] = []

    def agent_step(
        self,
        *,
        tool: str,
        query: str,
        output_class: type[BaseModel] | None = None,
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> PlanBuilderLite:
        """Add a step that uses the execution agent with a tool."""
        self._steps.append(
            AgentStep(tool=tool, query=query, output_class=output_class, condition=condition)
        )
        return self

    def tool_step(
        self,
        *,
        tool: str,
        inputs: dict[str, Any],
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> PlanBuilderLite:
        """Add a step that directly invokes a tool."""
        self._steps.append(ToolStep(tool=tool, inputs=inputs, condition=condition))
        return self

    def llm_step(
        self,
        *,
        query: str,
        output_class: type[BaseModel] | None = None,
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> PlanBuilderLite:
        """Add a step that directly queries the LLM tool."""
        self._steps.append(LLMStep(query=query, output_class=output_class, condition=condition))
        return self

    def hook(
        self,
        hook: Callable[..., bool],
        inputs: dict[str, Any] | None = None,
        condition: str | Callable[[list[Any]], bool] | None = None,
    ) -> PlanBuilderLite:
        """Add a hook step.

        The callable receives the list of previous outputs as the first argument,
        followed by any templated inputs as keyword arguments. It should return ``True``
        to continue execution or ``False`` to stop execution.

        Args:
            hook: The callable to execute
            inputs: Optional dictionary of inputs that will be templated with $outputX
                   placeholders and passed as keyword arguments to the hook
            condition: Optional condition to check before running the step

        """
        self._steps.append(Hook(hook, inputs, condition))
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
        """Initialize the Portia Lite client.

        Args:
            config (Config): The configuration to initialize the Portia Lite client. If not
                provided, the
                default configuration will be used.
            tools (ToolRegistry | list[Any]): The registry or list of tools to use. If not
                provided, the open source tool registry will be used, alongside the default
                tools
                from Portia cloud if a Portia API key is set.

        """
        self.config = config if config else Config.from_default()
        logger_manager.configure_from_config(self.config)
        logger().info(f"Starting Portia Lite v{get_version()}")
        if self.config.portia_api_key and self.config.portia_api_endpoint:
            logger().info(f"Using Portia cloud API endpoint: {self.config.portia_api_endpoint}")
        self._log_models(self.config)
        # TODO: add telemetry
        if not self.config.has_api_key("portia_api_key"):
            logger().warning(
                "No Portia API key found, Portia cloud tools and storage will not be available.",
            )

        if isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        elif isinstance(tools, list):
            self.tool_registry = ToolRegistry(tools)
        else:
            self.tool_registry = DefaultToolRegistry(self.config)

        match self.config.storage_class:
            case StorageClass.MEMORY:
                self.storage = InMemoryStorage()
            case StorageClass.DISK:
                self.storage = DiskFileStorage(storage_dir=self.config.storage_dir)
            case StorageClass.CLOUD:
                self.storage = PortiaCloudStorage(config=self.config)

        self.end_user = EndUser(external_id="portia:lite_user")

        # Current execution state
        self.current_plan: Plan | None = None
        self.current_plan_run: PlanRun | None = None

    @traceable(name="Portia Run")
    def run(self, plan: Sequence[Step]) -> Any:
        """Execute the provided plan and return the final output."""
        # Create a Plan object from the steps for the PlanRun
        portia_plan = self._create_plan_from_steps(plan, "PortiaLite execution")
        self.save(portia_plan)

        # Create a PlanRun to track execution
        plan_run = PlanRun(
            plan_id=portia_plan.id,
            end_user_id=self.end_user.external_id,
            state=PlanRunState.IN_PROGRESS,
        )

        # Store instances for reuse in condition evaluation
        self.current_plan = portia_plan
        self.current_plan_run = plan_run

        # Save the initial plan run
        self.storage.save_plan_run(plan_run)
        logger().info(f"Created PlanRun {plan_run.id} for execution")

        outputs: list[Any] = []
        step_output_index = 0  # Track outputs that correspond to actual plan steps

        for i, step in enumerate(plan):
            logger().info(f"Starting step {i}: {step.describe()}")

            # Check condition before executing step
            if not self._should_execute_step(step, outputs):
                logger().info(f"Skipping step {i} due to condition")
                continue

            cont = step.run(self, outputs)

            # Update plan run with step output (if the step produces output)
            if outputs and len(outputs) > len(plan_run.outputs.step_outputs):
                output_key = f"$output_{step_output_index}"
                plan_run.outputs.step_outputs[output_key] = LocalDataValue(value=outputs[-1])
                step_output_index += 1
                logger().info(f"Completed step {i}: Result = {outputs[-1]}")
            else:
                logger().info(f"Completed step {i}: No output")

            # Save updated plan run after each step
            self.storage.save_plan_run(plan_run)

            if not cont:
                logger().info(f"Step {i} requested execution to stop")
                break

            if step.to_portia_step(i) is not None:
                plan_run.current_step_index += 1

        # Mark plan run as complete and set final output
        final_result = outputs[-1] if outputs else None
        plan_run.state = PlanRunState.COMPLETE
        if final_result is not None:
            plan_run.outputs.final_output = LocalDataValue(value=final_result)

        # Save final plan run state
        self.storage.save_plan_run(plan_run)
        logger().info(f"PlanRun {plan_run.id} completed with state {plan_run.state}")

        # Clear stored instances
        self.current_plan = None
        self.current_plan_run = None

        return final_result

    def save(self, plan: Sequence[Step] | Plan, query: str = "PortiaLite query") -> Plan:
        """Save a plan built with PlanBuilderLite to storage.

        Args:
            plan: The list of steps from PlanBuilderLite.build()
            query: The query description for the plan context

        Returns:
            Plan: The saved Plan object

        """
        if isinstance(plan, Sequence):
            portia_plan = self._create_plan_from_steps(plan, query)
        else:
            portia_plan = plan

        # Save to storage
        self.storage.save_plan(portia_plan)
        logger().info(
            f"Plan saved with {len(portia_plan.steps)} steps original steps)",
            plan=str(portia_plan.id),
        )

        return portia_plan

    def _create_plan_from_steps(
        self, plan: Sequence[Step], query: str = "PortiaLite query"
    ) -> Plan:
        """Create a Plan object from a sequence of steps.

        Args:
            plan: The list of steps from PlanBuilderLite.build()
            query: The query description for the plan context

        Returns:
            Plan: The created Plan object (not saved to storage)

        """
        # Convert each step to a PlanStep, filtering out None values
        plan_steps = []
        step_index = 0
        for step in plan:
            portia_step = step.to_portia_step(step_index)
            if portia_step is not None:
                plan_steps.append(portia_step)
                step_index += 1

        # Get all tool IDs from the steps
        tool_ids = [step.tool_id for step in plan_steps if step.tool_id is not None]

        # Create the Plan object
        return Plan(
            plan_context=PlanContext(query=query, tool_ids=tool_ids),
            steps=plan_steps,
            plan_inputs=[],  # Empty as specified
            structured_output_schema=None,  # Empty as specified
        )

    @traceable(name="Portia Condition Evaluation")
    def _should_execute_step(self, step: Step, outputs: list[Any]) -> bool:
        """Check if a step should be executed based on its condition.

        Args:
            step: The step to check
            outputs: List of previous step outputs

        Returns:
            bool: True if the step should be executed, False otherwise

        """
        if step.condition is None:
            return True

        if callable(step.condition):
            # If condition is a callable, call it with outputs
            try:
                return step.condition(outputs)
            except Exception as e:  # noqa: BLE001
                logger().error(f"Error evaluating callable condition: {e}")
                return False
        elif isinstance(step.condition, str):
            # If condition is a string, use introspection agent
            return self._evaluate_string_condition(step.condition, outputs)
        else:
            logger().error(f"Invalid condition type: {type(step.condition)}")
            return False

    def _evaluate_string_condition(self, condition: str, outputs: list[Any]) -> bool:
        """Evaluate a string condition using the introspection agent.

        Args:
            condition: The condition string to evaluate
            outputs: List of previous step outputs

        Returns:
            bool: True if the condition evaluates to continue, False otherwise

        Raises:
            RuntimeError: If called outside of an active execution context

        """
        if not self.current_plan or not self.current_plan_run:
            raise RuntimeError(
                "String condition evaluation requires an active execution context. "
                "_current_plan and _current_plan_run must be set."
            )

        try:
            logger().debug(f"Evaluating string condition: '{condition}'")
            introspection_agent = DefaultIntrospectionAgent(self.config, self.storage)
            pre_step_outcome = introspection_agent.pre_step_introspection(
                plan=ReadOnlyPlan.from_plan(self.current_plan),
                plan_run=ReadOnlyPlanRun.from_plan_run(self.current_plan_run),
            )
            result = pre_step_outcome.outcome == PreStepIntrospectionOutcome.CONTINUE
            logger().debug(f"String condition '{condition}' evaluated to: {result}")
        except Exception as e:  # noqa: BLE001
            logger().error(f"Error evaluating string condition '{condition}': {e}")
            return False
        else:
            return result

    @staticmethod
    def _log_models(config: Config) -> None:
        """Log the models set in the configuration."""
        logger().debug("Portia Generative Models")
        for model in GenerativeModelsConfig.model_fields:
            getter = getattr(config, f"get_{model}")
            logger().debug(f"{model}: {getter()}")
