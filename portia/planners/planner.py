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

from openai import BaseModel
from pydantic import ConfigDict, Field

from portia.plan import Plan, Step  # noqa: TC001

if TYPE_CHECKING:
    from portia.config import Config
    from portia.execution_context import ExecutionContext
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
    def generate_steps_or_error(
        self,
        ctx: ExecutionContext,
        query: str,
        tool_list: list[Tool],
        examples: list[Plan] | None = None,
    ) -> StepsOrError:
        """Generate a list of steps for the given query.

        This method should be implemented to generate a list of steps to accomplish the query based
        on the provided query and tools.

        Args:
            ctx (ExecutionContext): The context for execution.
            query (str): The user query to generate a list of steps for.
            tool_list (list[Tool]): A list of tools available for the plan.
            examples (list[Plan] | None): Optional list of example plans to guide the planner.

        Returns:
            StepsOrError: A StepsOrError instance containing either the generated steps or an error.

        """
        raise NotImplementedError("generate_steps_or_error is not implemented")


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
