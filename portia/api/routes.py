"""API routes for the Portia Cloud backend."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from portia.api.auth import AuthenticatedUser, check_plan_access, get_current_user
from portia.api.database import DatabaseClient, get_database
from portia.api.models import ErrorResponse, PlanResponse
from portia.errors import PlanNotFoundError
from portia.prefixed_uuid import PlanUUID

router = APIRouter(prefix="/v1", tags=["plans"])


@router.get(
    "/plans/{plan_id}",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Plan retrieved successfully",
            "model": PlanResponse,
        },
        401: {
            "description": "Unauthorized - Invalid or missing API key",
            "model": ErrorResponse,
        },
        403: {
            "description": "Forbidden - User does not have access to this plan",
            "model": ErrorResponse,
        },
        404: {
            "description": "Not Found - Plan does not exist",
            "model": ErrorResponse,
        },
    },
    summary="Retrieve a single plan by ID",
    description="""
    Retrieve a single Plan object by its unique ID from the Portia Cloud database.

    This endpoint requires authentication via API key and enforces organization-based
    authorization. Users can only access plans that belong to their organization.

    **Authentication:**
    - Requires a valid API key in the Authorization header
    - Format: `Authorization: Api-Key <your-api-key>`

    **Authorization:**
    - Users can only access plans within their own organization
    - Attempting to access a plan from another organization returns 403 Forbidden

    **Response:**
    The endpoint returns the complete plan details including:
    - The user's input query
    - The list of tools used
    - The complete sequence of generated steps
    - For each step: task, tool used, input, and output information
    """,
)
async def get_plan(
    plan_id: str,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    db: Annotated[DatabaseClient, Depends(get_database)],
) -> PlanResponse:
    """Retrieve a plan by its ID.

    This endpoint implements the GET /v1/plans/{plan_id} functionality with:
    - User authentication via API key
    - Organization-based authorization
    - Error handling for 404 (not found) and 403 (forbidden)

    Args:
        plan_id (str): The unique plan ID (format: plan_...)
        current_user (AuthenticatedUser): The authenticated user (injected)
        db (DatabaseClient): The database client (injected)

    Returns:
        PlanResponse: The complete plan details in JSON format

    Raises:
        HTTPException: 401 if authentication fails
        HTTPException: 403 if user doesn't have access to the plan
        HTTPException: 404 if the plan doesn't exist

    """
    # Parse and validate the plan ID format
    try:
        plan_uuid = PlanUUID.from_string(plan_id)
    except (ValueError, AttributeError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid plan ID format: {plan_id}",
        ) from e

    # Retrieve the plan from the database
    try:
        plan, organization_id = await db.get_plan_by_id(plan_uuid)
    except PlanNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan not found: {plan_id}",
        ) from e

    # Check authorization - ensure user's org matches plan's org
    check_plan_access(organization_id, current_user)

    # Return the plan response
    return PlanResponse(
        id=str(plan.id),
        query=plan.plan_context.query,
        tool_ids=plan.plan_context.tool_ids,
        steps=plan.steps,
        plan_inputs=plan.plan_inputs,
        organization_id=organization_id,
    )
