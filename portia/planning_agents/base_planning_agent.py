"""PlanningAgents module creates plans from queries.

This module contains the PlanningAgent interfaces and implementations used for generating plans
based on user queries. It supports the creation of plans using tools and example plans, and
leverages LLMs to generate detailed step-by-step plans. It also handles errors gracefully and
provides feedback in the form of error messages when the plan cannot be created.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from portia.plan import Plan, PlanInput, Step

if TYPE_CHECKING:
    from portia.config import Config
    from portia.end_user import EndUser
    from portia.tool import Tool

logger = logging.getLogger(__name__)


class BasePlanningAgent(ABC):
    """Interface for planning.

    This class defines the interface for PlanningAgents that generate plans based on queries.
    A PlanningAgent will implement the logic to generate a plan or an error given a query,
    a list of tools, and optionally, some example plans.

    Attributes:
        config (Config): Configuration settings for the PlanningAgent.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the PlanningAgent with configuration.

        Args:
            config (Config): The configuration to initialize the PlanningAgent.

        """
        self.config = config

    @abstractmethod
    def generate_steps_or_error(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        examples: list[Plan] | None = None,
        plan_inputs: list[PlanInput] | None = None,
    ) -> StepsOrError:
        """Generate a list of steps for the given query.

        This method should be implemented to generate a list of steps to accomplish the query based
        on the provided query and tools.

        Args:
            query (str): The user query to generate a list of steps for.
            tool_list (list[Tool]): A list of tools available for the plan.
            end_user (EndUser): The end user for this plan
            examples (list[Plan] | None): Optional list of example plans to guide the PlanningAgent.
            plan_inputs (list[PlanInput] | None): Optional list of PlanInput objects defining
                the inputs required for the plan.

        Returns:
            StepsOrError: A StepsOrError instance containing either the generated steps or an error.

        """
        raise NotImplementedError("generate_steps_or_error is not implemented")

    async def agenerate_steps_or_error(
        self,
        query: str,
        tool_list: list[Tool],
        end_user: EndUser,
        examples: list[Plan] | None = None,
        plan_inputs: list[PlanInput] | None = None,
    ) -> StepsOrError:
        """Generate a list of steps for the given query asynchronously.

        This method should be implemented to generate a list of steps to accomplish the query based
        on the provided query and tools.

        Args:
            query (str): The user query to generate a list of steps for.
            tool_list (list[Tool]): A list of tools available for the plan.
            end_user (EndUser): The end user for this plan
            examples (list[Plan] | None): Optional list of example plans to guide the PlanningAgent.
            plan_inputs (list[PlanInput] | None): Optional list of PlanInput objects defining
                the inputs required for the plan.

        Returns:
            StepsOrError: A StepsOrError instance containing either the generated steps or an error.

        """
        raise NotImplementedError("async is not implemented")  # pragma: no cover


class StepsOrError(BaseModel):
    """A list of steps or an error.

    This model represents either a list of steps for a plan or an error message if
    the steps could not be created.

    Attributes:
        steps (list[Step]): The generated steps if successful.
        error (str | None): An error message if the steps could not be created.

    """

    model_config = ConfigDict(extra="forbid")

    steps: list[Step]
    error: str | None = Field(
        default=None,
        description="An error message if the steps could not be created.",
    )
