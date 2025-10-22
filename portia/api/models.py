"""API models for the Portia Cloud API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from portia.plan import PlanInput, Step


class PlanResponse(BaseModel):
    """Response model for a single plan retrieved from the API.

    This model represents the full plan details returned by the GET /v1/plans/{plan_id}
    endpoint. It includes all information needed to reconstruct a Plan object on the
    client side.

    """

    id: str = Field(description="The unique plan ID (e.g., plan_...)")
    query: str = Field(description="The original user query that generated this plan")
    tool_ids: list[str] = Field(description="List of tool IDs available when the plan was created")
    steps: list[Step] = Field(description="The sequence of steps in the plan")
    plan_inputs: list[PlanInput] = Field(
        default=[],
        description="The inputs required by the plan",
    )
    organization_id: str = Field(
        description="The organization ID that owns this plan",
        exclude=True,  # Don't expose in the API response
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "id": "plan_abc123def456",
                "query": "Find the cheapest flights from NYC to LAX",
                "tool_ids": ["portia:web_search", "portia:flight_search"],
                "steps": [
                    {
                        "task": "Search for flights from NYC to LAX",
                        "tool_id": "portia:flight_search",
                        "output": "$flight_results",
                        "inputs": [],
                    },
                    {
                        "task": "Find the cheapest option",
                        "tool_id": None,
                        "output": "$cheapest_flight",
                        "inputs": [
                            {"name": "$flight_results", "description": "Flight search results"}
                        ],
                    },
                ],
                "plan_inputs": [],
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(description="Error message describing what went wrong")

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "detail": "Plan not found",
            }
        }
