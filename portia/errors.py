"""Central definition of error classes.

This module defines custom exception classes used throughout the application. These exceptions
help identify specific error conditions, particularly related to configuration, planning, runs,
tools, and storage. They provide more context and clarity than generic exceptions.

Classes in this file include:

- `ConfigNotFoundError`: Raised when a required configuration value is not found.
- `InvalidConfigError`: Raised when a configuration value is invalid.
- `PlanError`: A base class for exceptions in the query planning_agent module.
- `PlanNotFoundError`: Raised when a plan is not found.
- `PlanRunNotFoundError`: Raised when a PlanRun is not found.
- `ToolNotFoundError`: Raised when a tool is not found.
- `DuplicateToolError`: Raised when a tool is registered with the same name.
- `InvalidToolDescriptionError`: Raised when a tool description is invalid.
- `ToolRetryError`: Raised when a tool fails after retries.
- `ToolFailedError`: Raised when a tool fails with a hard error.
- `InvalidPlanRunStateError`: Raised when a plan run is in an invalid state.
- `InvalidAgentOutputError`: Raised when the agent produces invalid output.
- `ToolHardError`: Raised when a tool encounters an unrecoverable error.
- `ToolSoftError`: Raised when a tool encounters an error that can be retried.
- `StorageError`: Raised when an issue occurs with storage.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portia.plan import PlanUUID
    from portia.plan_run import PlanRunUUID


class PortiaBaseError(Exception):
    """Base class for all our errors."""


class SkipExecutionError(PortiaBaseError):
    """Raised when a Portia execution should be stopped or a step should be skipped."""

    # If True, the execution should be stopped and the plan run should be returned
    # If False, only the step should be skipped
    should_return: bool

    def __init__(self, reason: str, should_return: bool = False) -> None:
        """Set custom error message.

        Args:
            reason (str): The reason for skipping the step.
            should_return (bool): Whether to return the plan run and stop execution entirely,
                or just skip the step.

        """
        self.should_return = should_return
        super().__init__(f"Skipping step: {reason}")


class ConfigNotFoundError(PortiaBaseError):
    """Raised when a required configuration value is not found.

    Args:
        value (str): The name of the configuration value that is missing.

    """

    def __init__(self, value: str) -> None:
        """Set custom error message."""
        super().__init__(f"Config value {value} is not set")


class InvalidConfigError(PortiaBaseError):
    """Raised when a configuration value is invalid.

    Args:
        value (str): The name of the invalid configuration value.
        issue (str): A description of the issue with the configuration value.

    """

    def __init__(self, value: str, issue: str) -> None:
        """Set custom error message."""
        self.message = f"Config value {value.upper()} is not valid - {issue}"
        super().__init__(self.message)


class PlanError(PortiaBaseError):
    """Base class for exceptions in the query planning_agent module.

    This exception indicates an error that occurred during the planning phase.

    Args:
        error_string (str): A description of the error encountered during planning.

    """

    def __init__(self, error_string: str) -> None:
        """Set custom error message."""
        super().__init__(f"Error during planning: {error_string}")


class PlanNotFoundError(PortiaBaseError):
    """Raised when a plan with a specific ID is not found.

    Args:
        plan_id (PlanUUID): The ID of the plan that was not found.

    """

    def __init__(self, plan_id: PlanUUID) -> None:
        """Set custom error message."""
        super().__init__(f"Plan with id {plan_id!s} not found.")


class PlanRunNotFoundError(PortiaBaseError):
    """Raised when a PlanRun with a specific ID is not found.

    Args:
        plan_run_id (UUID | str | None): The ID or name of the PlanRun that was not found.

    """

    def __init__(self, plan_run_id: PlanRunUUID | str | None) -> None:
        """Set custom error message."""
        super().__init__(f"Run with id {plan_run_id!s} not found.")


class ToolNotFoundError(PortiaBaseError):
    """Raised when a tool with a specific ID is not found.

    Args:
        tool_id (str): The ID of the tool that was not found.

    """

    def __init__(self, tool_id: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool with id {tool_id} not found.")


class DuplicateToolError(PortiaBaseError):
    """Raised when a tool is registered with the same name.

    Args:
        tool_id (str): The ID of the tool that already exists.

    """

    def __init__(self, tool_id: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool with id {tool_id} already exists.")


class InvalidToolDescriptionError(PortiaBaseError):
    """Raised when a tool description is invalid.

    Args:
        tool_id (str): The ID of the tool with an invalid description.

    """

    def __init__(self, tool_id: str) -> None:
        """Set custom error message."""
        super().__init__(f"Invalid Description for tool with id {tool_id}")


class ToolRetryError(PortiaBaseError):
    """Raised when a tool fails after retrying.

    Args:
        tool_id (str): The ID of the tool that failed.
        error_string (str): A description of the error that occurred.

    """

    def __init__(self, tool_id: str, error_string: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool {tool_id} failed after retries: {error_string}")


class ToolFailedError(PortiaBaseError):
    """Raised when a tool fails with a hard error.

    Args:
        tool_id (str): The ID of the tool that failed.
        error_string (str): A description of the error that occurred.

    """

    def __init__(self, tool_id: str, error_string: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool {tool_id} failed: {error_string}")


class InvalidPlanRunStateError(PortiaBaseError):
    """Raised when a plan run is in an invalid state."""


class InvalidAgentError(PortiaBaseError):
    """Raised when an agent is in an invalid state."""

    def __init__(self, state: str) -> None:
        """Set custom error message."""
        super().__init__(f"Agent returned invalid state: {state}")


class InvalidAgentOutputError(PortiaBaseError):
    """Raised when the agent produces invalid output.

    Args:
        content (str): The invalid content returned by the agent.

    """

    def __init__(self, content: str) -> None:
        """Set custom error message."""
        super().__init__(f"Agent returned invalid content: {content}")


class ToolHardError(PortiaBaseError):
    """Raised when a tool encounters an error it cannot retry.

    Args:
        cause (Exception | str): The underlying exception or error message.

    """

    def __init__(self, cause: Exception | str) -> None:
        """Set custom error message."""
        super().__init__(cause)


class ToolSoftError(PortiaBaseError):
    """Raised when a tool encounters an error that can be retried.

    Args:
        cause (Exception | str): The underlying exception or error message.

    """

    def __init__(self, cause: Exception | str) -> None:
        """Set custom error message."""
        super().__init__(cause)


class StorageError(PortiaBaseError):
    """Raised when there's an issue with storage.

    Args:
        cause (Exception | str): The underlying exception or error message.

    """

    def __init__(self, cause: Exception | str) -> None:
        """Set custom error message."""
        super().__init__(cause)


class PlanRunExitError(PortiaBaseError):
    """Raised when a step ascertains that a plan run should be exited.

    Args:
        reason (str): The reason for exiting the plan run.

    """

    def __init__(self, reason: str) -> None:
        """Set custom error message."""
        super().__init__(reason)
