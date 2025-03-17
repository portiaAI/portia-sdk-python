"""The default introspection agent.

This agent looks at the state of a plan run between steps
and makes decisions about whether execution should continue.
"""

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

from portia.config import INTROSPECTION_MODEL_KEY, Config
from portia.introspection_agents.introspection_agent import (
    BaseIntrospectionAgent,
    PreStepIntrospection,
)
from portia.llm_wrapper import LLMWrapper
from portia.plan import Plan
from portia.plan_run import PlanRun


class DefaultIntrospectionAgent(BaseIntrospectionAgent):
    """Default Introspection Agent.

    Implements the BaseIntrospectionAgent interface using an LLM to make decisions about what to do.

    Attributes:
        config (Config): Configuration settings for the DefaultIntrospectionAgent.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the DefaultIntrospectionAgent with configuration.

        Args:
            config (Config): The configuration to initialize the DefaultIntrospectionAgent.

        """
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a highly skilled reviewer who reviews in flight plan execution."
                        "Your job is to examine the state of a plan execution (PlanRun) and "
                        "decide what action should be taken next."
                        "You should use the current_step_index field to identify the current step"
                        "in the plan, and the PlanRun state to know what has happened so far."
                        "The actions that can be taken next in priority order are:"
                        " - STOP -> stop execution and return the result so far."
                        " - SKIP -> skip the next step execution."
                        " - FAIL -> stop execution entirely."
                        " - CONTINUE -> Continue execution of the next step."
                        "You should choose an outcome based on the following logic in order:\n"
                        " - If all remaining steps can not be executed because of the"
                        "   current conditions are not met return STOP.\n"
                        " - If the current step has a condition that is false you return SKIP.\n"
                        " - If you cannot evaluate the condition because some data had been skipped"
                        "  in previous steps then return SKIP.\n"
                        " - If you cannot evaluate the condition"
                        " because of missing data return FAIL.\n"
                        " - Otherwise return CONTINUE.\n"
                        "Return the outcome and reason in the given format.\n"
                    ),
                ),
                HumanMessagePromptTemplate.from_template(
                    "Review the following plan + current PlanRun."
                    "Current Plan: {plan}\n"
                    "Current PlanRun: {plan_run}\n",
                ),
            ],
        )

    def pre_step_introspection(
        self,
        plan: Plan,
        plan_run: PlanRun,
    ) -> PreStepIntrospection:
        """Ask the LLM whether to continue, skip or fail the plan_run."""
        outcome = (
            LLMWrapper.for_usage(INTROSPECTION_MODEL_KEY, self.config)
            .to_langchain()
            .with_structured_output(PreStepIntrospection)
            .invoke(
                self.prompt.format_messages(
                    plan_run=plan_run.model_dump_json(),
                    plan=plan.model_dump_json(),
                ),
            )
        )

        return PreStepIntrospection.model_validate(outcome)
