"""Planner module creates plans from queries.

This module contains the planner interfaces and implementations used for generating plans
based on user queries. It supports the creation of plans using tools and example plans, and
leverages LLMs to generate detailed step-by-step plans. It also handles errors gracefully and
provides feedback in the form of error messages when the plan cannot be created.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import Field

from portia.common import PortiaBaseModel

if TYPE_CHECKING:
    from portia.config import Config
    from portia.execution_context import ExecutionContext
    from portia.plan import Plan, Step
    from portia.tool import Tool

logger = logging.getLogger(__name__)


class Planner(ABC):
    """Interface for planning.

    This class defines the interface for planners that generate plans based on queries.
    A planner will implement the logic to generate a plan or an error given a query,
    a list of tools, and optionally, some example plans.

    Attributes:
        config (Config): Configuration settings for the planner.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the planner with configuration.

        Args:
            config (Config): The configuration to initialize the planner.

        """
        self.config = config

    @abstractmethod
    def generate_plan_or_error(
        self,
        ctx: ExecutionContext,
        query: str,
        tool_list: list[Tool],
        examples: list[Plan] | None = None,
    ) -> PlanOrError:
        """Generate a plan for the given query.

        This method should be implemented to generate a detailed plan of action based on
        the provided query and tools.

        Args:
            ctx (ExecutionContext): The context for execution.
            query (str): The user query to generate a plan for.
            tool_list (list[Tool]): A list of tools available for the plan.
            examples (list[Plan] | None): Optional list of example plans to guide the planner.

        Returns:
            PlanOrError: A PlanOrError instance containing either the generated plan or an error.

        """
        raise NotImplementedError("generate_plan_or_error is not implemented")


# TODO(Emma): This is a temporary class while we are migrating to a synced plan model. #noqa: FIX002
# Evals should be updated to use the new StepsOrError class.
# https://linear.app/portialabs/issue/POR-381
class PlanOrError(PortiaBaseModel):
    """A plan or an error.

    This model represents either a successful plan or an error message if the plan could
    not be created.

    Attributes:
        plan (Plan): The generated plan if successful.
        error (str | None): An error message if the plan could not be created.

    """

    plan: Plan
    error: str | None = Field(
        default=None,
        description="An error message if the plan could not be created.",
    )


class StepsOrError(PortiaBaseModel):
    """A list of steps or an error.

    This model represents either a list of steps for a plan or an error message if
    the steps could not be created.

    Attributes:
        steps (list[Step]): The generated steps if successful.
        error (str | None): An error message if the steps could not be created.

    """

    steps: list[Step]
    error: str | None = Field(
        default=None,
        description="An error message if the steps could not be created.",
    )
