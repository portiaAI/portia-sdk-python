"""Authentication and authorization utilities for the Portia Cloud API."""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


class AuthenticatedUser:
    """Represents an authenticated user with organization information."""

    def __init__(self, user_id: str, organization_id: str, api_key: str) -> None:
        """Initialize an authenticated user.

        Args:
            user_id (str): The unique identifier for the user.
            organization_id (str): The unique identifier for the user's organization.
            api_key (str): The API key used for authentication.

        """
        self.user_id = user_id
        self.organization_id = organization_id
        self.api_key = api_key


async def get_current_user(
    authorization: str | None = Security(api_key_header),
) -> AuthenticatedUser:
    """Validate the API key and return the authenticated user.

    This function validates the API key from the Authorization header and returns
    an AuthenticatedUser object with the user's information. In a real implementation,
    this would query a database to validate the key and fetch user information.

    Args:
        authorization (str | None): The Authorization header value.

    Returns:
        AuthenticatedUser: The authenticated user information.

    Raises:
        HTTPException: If the API key is missing or invalid (401 Unauthorized).

    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Expected format: "Api-Key <key>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0] != "Api-Key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication format. Expected 'Api-Key <key>'",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    api_key = parts[1]

    # TODO: In production, validate the API key against the database
    # For now, we'll simulate a successful authentication
    # This is where you would:
    # 1. Query the database to find the API key
    # 2. Check if the key is valid and not expired
    # 3. Fetch the associated user and organization information
    # 4. Return the AuthenticatedUser object

    # Simulated validation (replace with actual database query)
    if not api_key or len(api_key) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # In a real implementation, fetch these from the database based on the API key
    # For now, we'll extract from a mock format or use placeholder values
    return AuthenticatedUser(
        user_id="user_123",  # Would come from database
        organization_id="org_456",  # Would come from database
        api_key=api_key,
    )


def check_plan_access(plan_owner_org_id: str, current_user: AuthenticatedUser) -> None:
    """Check if the current user has access to a plan.

    This function verifies that the current user's organization matches the
    organization that owns the plan. If they don't match, a 403 Forbidden
    error is raised.

    Args:
        plan_owner_org_id (str): The organization ID that owns the plan.
        current_user (AuthenticatedUser): The currently authenticated user.

    Raises:
        HTTPException: If the user's organization doesn't match (403 Forbidden).

    """
    if current_user.organization_id != plan_owner_org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this plan",
        )
