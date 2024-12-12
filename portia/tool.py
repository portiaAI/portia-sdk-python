"""Tools module.

This module defines an abstract base class for tools that can be extended to create custom tools
Each tool has a unique ID and a name, and child classes should implement the `run` method
with their specific logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from uuid import uuid4

Serializable = Any
SERIALIZABLE_TYPE_VAR = TypeVar("SERIALIZABLE_TYPE_VAR", bound=Serializable)


class Tool(ABC, Generic[SERIALIZABLE_TYPE_VAR]):
    """Abstract base class for a tool.

    This class serves as the blueprint for all tools. Child classes must implement the `run` method.

    Attributes:
        id (str): A unique identifier generated for the tool.
        name (str): The name of the tool.

    """

    def __init__(self, name: str) -> None:
        """Initialize a new tool instance.

        Args:
            name (str): The name of the tool.

        """
        self.id = str(uuid4())
        self.name = name

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
