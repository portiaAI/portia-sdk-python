"""Tools module.

This module defines an abstract base class for tools that can be extended to create custom tools
Each tool has a unique ID and a name, and child classes should implement the `run` method
with their specific logic.
"""

from abc import abstractmethod
from typing import Any, Generic

from pydantic import BaseModel, Field

from portia.types import SERIALIZABLE_TYPE_VAR


class Tool(BaseModel, Generic[SERIALIZABLE_TYPE_VAR]):
    """Abstract base class for a tool.

    This class serves as the blueprint for all tools. Child classes must implement the `run` method.

    Attributes:
        id (str): A unique identifier for the tool.
        name (str): The name of the tool.
        description (str): Purpose of the tool and usage.

    """

    id: str = Field(description="ID of the tool")
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Purpose of the tool and usage")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> SERIALIZABLE_TYPE_VAR:  # noqa: ANN401
        """Run the tool.

        This method must be implemented by subclasses to define the tool's specific behavior.

        Args:
            args (Any): The arguments passed to the tool for execution.
            kwargs (Any): The keyword arguments passed to the tool for execution.

        Returns:
            Any: The result of the tool's execution.

        """
