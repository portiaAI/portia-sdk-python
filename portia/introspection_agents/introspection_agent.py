"""BaseIntrospectionAgent is the interface for all introspection agents."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from portia.common import PortiaEnum
from portia.config import Config
from portia.plan import Plan
from portia.plan_run import PlanRun


class PreStepIntrospectionOutcome(PortiaEnum):
    """The Outcome of the introspection."""

    CONTINUE = "CONTINUE"
    SKIP = "SKIP"
    FAIL = "FAIL"
    STOP = "STOP"


class PreStepIntrospection(BaseModel):
    """The outcome of a pre-step introspection."""

    outcome: PreStepIntrospectionOutcome = Field(
        default=PreStepIntrospectionOutcome.CONTINUE,
        description="What action should be taken next based on the state of the plan run.",
    )
    reason: str = Field(
        description="The reason the given outcome was decided on.",
    )


class BaseIntrospectionAgent(ABC):
    """Interface for introspection.

    This class defines the interface for introspection.
    By introspection we mean looking at the state of a plan run and making decisions
    about whether to continue.

    Attributes:
        config (Config): Configuration settings for the PlanningAgent.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the BaseIntrospectionAgent with configuration.

        Args:
            config (Config): The configuration to initialize the BaseIntrospectionAgent.

        """
        self.config = config

    @abstractmethod
    def pre_step_introspection(
        self,
        plan: Plan,
        plan_run: PlanRun,
    ) -> PreStepIntrospection:
        """pre_step_introspection is introspection run before a plan happens.."""
        raise NotImplementedError("pre_step_introspection is not implemented")
