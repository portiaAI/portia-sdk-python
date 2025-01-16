"""Central definition of error classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


class ConfigNotFoundError(Exception):
    """Raised when a needed config value is not present."""

    def __init__(self, value: str) -> None:
        """Set custom error message."""
        super().__init__(f"Config value {value} is not set")


class InvalidConfigError(Exception):
    """Raised when a needed config value is invalid."""

    def __init__(self, value: str, issue: str) -> None:
        """Set custom error message."""
        super().__init__(f"Config value {value.upper()} is not valid - {issue}")


class PlanError(Exception):
    """Base class for exceptions in the query planner module. Indicates an error in planning."""

    def __init__(self, error_string: str) -> None:
        """Set custom error message."""
        super().__init__(f"Error during planning: {error_string}")


class PlanNotFoundError(Exception):
    """Indicate a plan was not found."""

    def __init__(self, plan_id: UUID) -> None:
        """Set custom error message."""
        super().__init__(f"Plan with id {plan_id} not found.")


class WorkflowNotFoundError(Exception):
    """Indicate a workflow was not found."""

    def __init__(self, workflow_id: UUID | str | None) -> None:
        """Set custom error message."""
        super().__init__(f"Workflow with id {workflow_id} not found.")


class ToolNotFoundError(Exception):
    """Custom error class when tools aren't found."""

    def __init__(self, tool_name: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool with name {tool_name} not found.")


class DuplicateToolError(Exception):
    """Custom error class when tools are registered with the same name."""

    def __init__(self, tool_name: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool with name {tool_name} already exists.")


class InvalidToolDescriptionError(Exception):
    """Raised when the tool description is invalid."""

    def __init__(self, tool_name: str) -> None:
        """Set custom error message."""
        super().__init__(f"Invalid Description for tool with name {tool_name}")


class ToolRetryError(Exception):
    """Raised when a tool fails on a retry."""

    def __init__(self, tool_name: str, error_string: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool {tool_name} failed after retries: {error_string}")


class ToolFailedError(Exception):
    """Raised when a tool fails with a hard error."""

    def __init__(self, tool_name: str, error_string: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool {tool_name} failed: {error_string}")


class InvalidWorkflowStateError(Exception):
    """The given workflow is in an invalid state."""


class InvalidAgentOutputError(Exception):
    """The agent returned output that could not be processed."""

    def __init__(self, content: str) -> None:
        """Set custom error message."""
        super().__init__(f"Agent returned invalid content: {content}")


class ToolHardError(Exception):
    """Raised when a tool hits an error it can't retry."""

    def __init__(self, cause: Exception | str) -> None:
        """Set custom error message."""
        super().__init__(cause)


class ToolSoftError(Exception):
    """Raised when a tool hits an error it can retry."""

    def __init__(self, cause: Exception | str) -> None:
        """Set custom error message."""
        super().__init__(cause)


class StorageError(Exception):
    """Raised when there's an issue with the storage."""

    def __init__(self, cause: Exception | str) -> None:
        """Set custom error message."""
        super().__init__(cause)
