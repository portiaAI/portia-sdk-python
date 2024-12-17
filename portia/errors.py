"""Central definition of error classes."""

from uuid import UUID


class InvalidStorageError(Exception):
    """Raised when an invalid storage is provided."""

    def __init__(self, storage: str) -> None:
        """Set custom error message."""
        super().__init__(f"{storage} is not a valid storage provider.")


class InvalidLLMProviderError(Exception):
    """Raised when a provider is invalid."""

    def __init__(self, provider: str) -> None:
        """Set custom error message."""
        super().__init__(f"{provider} is not a supported LLM provider")


class ConfigNotFoundError(Exception):
    """Raised when a needed config value is not present."""

    def __init__(self, value: str) -> None:
        """Set custom error message."""
        super().__init__(f"Config value {value} is not set")


class InvalidConfigError(Exception):
    """Raised when a needed config value is invalid."""

    def __init__(self, value: str) -> None:
        """Set custom error message."""
        super().__init__(f"Config value {value} is not valid")


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

    def __init__(self, workflow_id: UUID) -> None:
        """Set custom error message."""
        super().__init__(f"Workflow with id {workflow_id} not found.")


class ToolNotFoundError(Exception):
    """Custom error class when tools aren't found."""

    def __init__(self, tool_name: str) -> None:
        """Set custom error message."""
        super().__init__(f"Tool with name {tool_name} not found.")


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

    def __init__(self, workflow_id: UUID) -> None:
        """Set custom error message."""
        super().__init__(f"Workflow with id {workflow_id} is in invalid state.")


class InvalidAgentError(Exception):
    """Raise when a given agent is invalid."""

    def __init__(self, agent: str) -> None:
        """Set custom error message."""
        super().__init__(f"{agent} is not a supported agent")


class InvalidAgentUsageError(Exception):
    """Raise when a given agent is used incorrectly."""

    def __init__(self, agent: str) -> None:
        """Set custom error message."""
        super().__init__(f"{agent} is not being used in correct context")


class NoVerifiedArgsError(Exception):
    """Raised when verified args are expected but not present."""

    def __init__(self) -> None:
        """Set custom error message."""
        super().__init__("Expected verified args to be provided")
