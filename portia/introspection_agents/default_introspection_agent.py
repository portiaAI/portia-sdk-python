"""The default introspection agent.

This agent looks at the state of a plan run between steps
and makes decisions about whether execution should continue.
"""

from datetime import UTC, datetime

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

from portia.config import Config
from portia.introspection_agents.introspection_agent import (
    BaseIntrospectionAgent,
    PreStepIntrospection,
)
from portia.logger import logger
from portia.model import Message
from portia.plan import Plan
from portia.plan_run import PlanRun
from portia.storage import AgentMemory, AgentMemoryValue


class DefaultIntrospectionAgent(BaseIntrospectionAgent):
    """Default Introspection Agent.

    Implements the BaseIntrospectionAgent interface using an LLM to make decisions about what to do.

    Attributes:
        config (Config): Configuration settings for the DefaultIntrospectionAgent.

    """

    def __init__(self, config: Config, agent_memory: AgentMemory) -> None:
        """Initialize the DefaultIntrospectionAgent with configuration.

        Args:
            config (Config): The configuration to initialize the DefaultIntrospectionAgent.
            agent_memory (AgentMemory): The agent memory to use

        """
        self.config = config
        self.agent_memory = agent_memory

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        """
You are a highly skilled reviewer who reviews in flight plan execution. Your job is to evaluate
the condition for the current step. Your outcome is fed to another orchestrator that controls
the execution of the remaining steps.

IMPORTANT GUIDELINES:
- Pay close attention to the steps giving and its tasks, there is no alternative flows or other
steps other than what's been giving to you for this plan execution.
- Do not assume data, you should evaluate the condition ONLY based on data given.
- Do not assume if the condition is false it will affect other remaining steps. Step A will affect
step B ONLY if Step A output is included in the inputs list for Step B or
mentioned in the condition attribute for step B.
- Steps CAN NOT get executed on partial information. A step will only be executed if
all inputs are present and its condition is true (if the step has condition).
- Your outcome will not affect any future plan runs.

Provide an outcome from the following list (ordered by preference):
  1- COMPLETE -> stops the execution and does not execute remaining steps.
   - Choose COMPLETE if all remaining steps depend on the current step's output.
   - Choose COMPLETE if the condition of this step is the same for all remaining steps.
  2- SKIP -> ONLY skips the current step and continue executing next steps.
  3- CONTINUE -> continue execution for the current step.

You should evaluate the condition and provide the outcome based on the following criteria IN ORDER:
 1- If condition is false and all remaining steps depend on the output of this step
 then return COMPLETE.
 2- If condition is false and all remaining steps have the same condition return COMPLETE.
 3- If condition is false you return SKIP. But favour COMPLETE whenever possible
 (e.g if it is the last step).
 4- If you cannot evaluate the condition because some data had been skipped
  in previous steps then return SKIP.
 5- Otherwise return CONTINUE.

Return the outcome and reason in the given format.
"""
                    ),
                ),
                HumanMessagePromptTemplate.from_template(
                    "Today's date is {current_date} and today is {current_day_of_week}.\n"
                    "Review the following plan + current PlanRun.\n"
                    "The condition to evaluate is: {condition}.\n"
                    "We are at step {current_step_idex} out of {total_steps_count} in total.\n"
                    "The original query: {query}\n"
                    "All Plan Steps: \n{plan}\n"
                    "Previous Step Outputs: \n{prev_step_outputs}\n"
                    "Plan Run Inputs: \n{plan_run_inputs}\n"
                    "If any relevant outputs are stored in agent memory, they have been extracted "
                    "and included here: {memory_outputs}\n",
                ),
            ],
        )

    def pre_step_introspection(
        self,
        plan: Plan,
        plan_run: PlanRun,
    ) -> PreStepIntrospection:
        """Ask the LLM whether to continue, skip or fail the plan_run."""
        introspection_condition = plan.steps[plan_run.current_step_index].condition

        memory_outputs = [
            self.agent_memory.get_plan_run_output(output.output_name, plan_run.id)
            for output in plan_run.outputs.step_outputs.values()
            if isinstance(output, AgentMemoryValue)
            and introspection_condition
            and output.output_name in introspection_condition
        ]

        model = self.config.get_introspection_model()
        logger().trace("LLM call: introspection")
        return model.get_structured_response(
            schema=PreStepIntrospection,
            messages=[
                Message.from_langchain(m)
                for m in self.prompt.format_messages(
                    current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                    current_day_of_week=datetime.now(UTC).strftime("%A"),
                    prev_step_outputs=plan_run.outputs.model_dump_json(),
                    plan_run_inputs=plan_run.plan_run_inputs,
                    memory_outputs=memory_outputs,
                    query=plan.plan_context.query,
                    condition=plan.steps[plan_run.current_step_index].condition,
                    current_step_idex=plan_run.current_step_index + 1,
                    total_steps_count=len(plan.steps),
                    plan=self._get_plan_steps_pretty(plan),
                )
            ],
        )

    def _get_plan_steps_pretty(self, plan: Plan) -> str:
        """Get the pretty print representation of the plan steps."""
        return "\n".join(
            [f"Step {i+1}: {step.pretty_print()}" for i, step in enumerate(plan.steps)]
        )

    async def apre_step_introspection(
        self,
        plan: Plan,
        plan_run: PlanRun,
    ) -> PreStepIntrospection:
        """pre_step_introspection is introspection run before a plan happens.."""
        introspection_condition = plan.steps[plan_run.current_step_index].condition

        memory_outputs = [
            await self.agent_memory.aget_plan_run_output(output.output_name, plan_run.id)
            for output in plan_run.outputs.step_outputs.values()
            if isinstance(output, AgentMemoryValue)
            and introspection_condition
            and output.output_name in introspection_condition
        ]

        model = self.config.get_introspection_model()
        logger().trace("LLM call: introspection")
        return await model.aget_structured_response(
            schema=PreStepIntrospection,
            messages=[
                Message.from_langchain(m)
                for m in self.prompt.format_messages(
                    current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                    current_day_of_week=datetime.now(UTC).strftime("%A"),
                    prev_step_outputs=plan_run.outputs.model_dump_json(),
                    plan_run_inputs=plan_run.plan_run_inputs,
                    memory_outputs=memory_outputs,
                    query=plan.plan_context.query,
                    condition=plan.steps[plan_run.current_step_index].condition,
                    current_step_idex=plan_run.current_step_index + 1,
                    total_steps_count=len(plan.steps),
                    plan=self._get_plan_steps_pretty(plan),
                )
            ],
        )
