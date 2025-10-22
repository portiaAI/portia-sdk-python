"""Database access layer for the Portia Cloud API.

This module provides an interface for accessing plans from the database.
In a production environment, this would connect to the actual Portia Cloud
database (e.g., PostgreSQL, MongoDB, etc.).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from portia.errors import PlanNotFoundError

if TYPE_CHECKING:
    from portia.plan import Plan, PlanUUID


class DatabaseClient:
    """Client for accessing the Portia Cloud database.

    In a production implementation, this class would handle:
    - Database connection pooling
    - Query execution
    - Transaction management
    - Error handling

    For this implementation, we provide the interface that would integrate
    with the actual database backend.

    """

    def __init__(self) -> None:
        """Initialize the database client.

        In production, this would set up connection pools, etc.

        """
        # TODO: Initialize database connection pool

    async def get_plan_by_id(self, plan_id: PlanUUID) -> tuple[Plan, str]:
        """Retrieve a plan from the database by its ID.

        This method fetches a plan from the database and returns both the Plan
        object and the organization ID that owns it.

        Args:
            plan_id (PlanUUID): The unique identifier of the plan to retrieve.

        Returns:
            tuple[Plan, str]: A tuple containing the Plan object and the organization ID.

        Raises:
            PlanNotFoundError: If no plan exists with the given ID.

        """
        # TODO: In production, query the database
        # Example SQL query:
        # SELECT p.*, org.id as organization_id
        # FROM plans p
        # JOIN organizations org ON p.organization_id = org.id
        # WHERE p.id = $1
        #
        # For now, we'll raise a not found error as there's no database connected
        raise PlanNotFoundError(plan_id)

    async def plan_exists(self, plan_id: PlanUUID) -> bool:
        """Check if a plan exists in the database.

        Args:
            plan_id (PlanUUID): The unique identifier of the plan to check.

        Returns:
            bool: True if the plan exists, False otherwise.

        """
        try:
            await self.get_plan_by_id(plan_id)
            return True
        except PlanNotFoundError:
            return False


# Singleton database client instance
# In production, this would be properly initialized with configuration
db_client = DatabaseClient()


async def get_database() -> DatabaseClient:
    """Dependency injection function for FastAPI to get the database client.

    Returns:
        DatabaseClient: The database client instance.

    """
    return db_client
