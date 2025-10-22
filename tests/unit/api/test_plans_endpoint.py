"""Tests for the GET /v1/plans/{plan_id} endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from portia.api.app import app
from portia.api.auth import AuthenticatedUser, get_current_user
from portia.api.database import DatabaseClient, get_database
from portia.errors import PlanNotFoundError
from portia.plan import Plan, PlanContext, PlanUUID, Step

client = TestClient(app)


@pytest.fixture
def mock_authenticated_user() -> AuthenticatedUser:
    """Create a mock authenticated user for testing.

    Returns:
        AuthenticatedUser: A mock user with organization ID "org_123".

    """
    return AuthenticatedUser(
        user_id="user_456",
        organization_id="org_123",
        api_key="test_api_key_12345",
    )


@pytest.fixture
def sample_plan() -> Plan:
    """Create a sample plan for testing.

    Returns:
        Plan: A sample plan with test data.

    """
    return Plan(
        id=PlanUUID.from_string("plan-00000000-0000-0000-0000-000000000001"),
        plan_context=PlanContext(
            query="Find the best pizza places in New York",
            tool_ids=["portia:web_search", "portia:maps"],
        ),
        steps=[
            Step(
                task="Search for pizza places in New York",
                tool_id="portia:web_search",
                output="$search_results",
                inputs=[],
            ),
            Step(
                task="Find the highest rated option",
                tool_id=None,
                output="$best_pizza_place",
                inputs=[],
            ),
        ],
        plan_inputs=[],
    )


@pytest.fixture
def mock_db() -> AsyncMock:
    """Create a mock database client.

    Returns:
        AsyncMock: A mock DatabaseClient.

    """
    return AsyncMock(spec=DatabaseClient)


class TestGetPlanEndpoint:
    """Test suite for the GET /v1/plans/{plan_id} endpoint."""

    def test_get_plan_success(
        self,
        mock_authenticated_user: AuthenticatedUser,
        sample_plan: Plan,
        mock_db: AsyncMock,
    ) -> None:
        """Test successful plan retrieval with valid authentication and authorization.

        Verifies that:
        - A 200 OK status is returned
        - The response contains all plan details
        - The plan data matches the expected format

        """
        plan_id = str(sample_plan.id)
        organization_id = "org_123"  # Matches the user's org

        # Mock the database to return the plan
        mock_db.get_plan_by_id = AsyncMock(return_value=(sample_plan, organization_id))

        # Override FastAPI dependencies
        app.dependency_overrides[get_current_user] = lambda: mock_authenticated_user
        app.dependency_overrides[get_database] = lambda: mock_db

        try:
            response = client.get(
                f"/v1/plans/{plan_id}",
                headers={"Authorization": "Api-Key test_api_key_12345"},
            )
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["id"] == plan_id
        assert data["query"] == sample_plan.plan_context.query
        assert data["tool_ids"] == sample_plan.plan_context.tool_ids
        assert len(data["steps"]) == len(sample_plan.steps)
        assert data["steps"][0]["task"] == sample_plan.steps[0].task
        assert data["steps"][0]["tool_id"] == sample_plan.steps[0].tool_id
        assert data["steps"][0]["output"] == sample_plan.steps[0].output

    def test_get_plan_not_found(
        self,
        mock_authenticated_user: AuthenticatedUser,
        mock_db: AsyncMock,
    ) -> None:
        """Test that a 404 error is returned when the plan does not exist.

        Verifies that:
        - A 404 NOT FOUND status is returned
        - The error message indicates the plan was not found

        """
        plan_id = "plan-00000000-0000-0000-0000-000000000099"

        # Mock the database to raise PlanNotFoundError
        mock_db.get_plan_by_id = AsyncMock(
            side_effect=PlanNotFoundError(PlanUUID.from_string(plan_id))
        )

        # Override FastAPI dependencies
        app.dependency_overrides[get_current_user] = lambda: mock_authenticated_user
        app.dependency_overrides[get_database] = lambda: mock_db

        try:
            response = client.get(
                f"/v1/plans/{plan_id}",
                headers={"Authorization": "Api-Key test_api_key_12345"},
            )
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"].lower()
        assert plan_id in data["detail"]

    def test_get_plan_forbidden_different_organization(
        self,
        mock_authenticated_user: AuthenticatedUser,
        sample_plan: Plan,
        mock_db: AsyncMock,
    ) -> None:
        """Test that a 403 error is returned when accessing a plan from another organization.

        Verifies that:
        - A 403 FORBIDDEN status is returned
        - The error message indicates lack of permission

        """
        plan_id = str(sample_plan.id)
        organization_id = "org_999"  # Different from user's org (org_123)

        # Mock the database to return a plan from a different organization
        mock_db.get_plan_by_id = AsyncMock(return_value=(sample_plan, organization_id))

        # Override FastAPI dependencies
        app.dependency_overrides[get_current_user] = lambda: mock_authenticated_user
        app.dependency_overrides[get_database] = lambda: mock_db

        try:
            response = client.get(
                f"/v1/plans/{plan_id}",
                headers={"Authorization": "Api-Key test_api_key_12345"},
            )
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_403_FORBIDDEN
        data = response.json()
        assert "permission" in data["detail"].lower() or "forbidden" in data["detail"].lower()

    def test_get_plan_unauthorized_missing_api_key(self) -> None:
        """Test that a 401 error is returned when the API key is missing.

        Verifies that:
        - A 401 UNAUTHORIZED status is returned
        - The error message indicates missing authentication

        """
        plan_id = "plan_test123"

        response = client.get(f"/v1/plans/{plan_id}")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "api key" in data["detail"].lower() or "missing" in data["detail"].lower()

    def test_get_plan_unauthorized_invalid_api_key_format(self) -> None:
        """Test that a 401 error is returned when the API key format is invalid.

        Verifies that:
        - A 401 UNAUTHORIZED status is returned
        - The error message indicates invalid authentication format

        """
        plan_id = "plan_test123"

        # Test with invalid format (missing "Api-Key" prefix)
        response = client.get(
            f"/v1/plans/{plan_id}",
            headers={"Authorization": "Bearer invalid_token"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "invalid" in data["detail"].lower() or "format" in data["detail"].lower()

    def test_get_plan_unauthorized_invalid_api_key(self) -> None:
        """Test that a 401 error is returned when the API key is invalid.

        Verifies that:
        - A 401 UNAUTHORIZED status is returned
        - The error message indicates invalid API key

        """
        plan_id = "plan_test123"

        # Test with too short API key (fails validation)
        response = client.get(
            f"/v1/plans/{plan_id}",
            headers={"Authorization": "Api-Key short"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "invalid" in data["detail"].lower() or "api key" in data["detail"].lower()

    def test_get_plan_invalid_plan_id_format(
        self,
        mock_authenticated_user: AuthenticatedUser,
        mock_db: AsyncMock,
    ) -> None:
        """Test that a 400 error is returned when the plan ID format is invalid.

        Verifies that:
        - A 400 BAD REQUEST status is returned
        - The error message indicates invalid plan ID format

        """
        plan_id = "invalid_format"  # Missing "plan_" prefix and proper UUID format

        # Override FastAPI dependencies
        app.dependency_overrides[get_current_user] = lambda: mock_authenticated_user
        app.dependency_overrides[get_database] = lambda: mock_db

        try:
            response = client.get(
                f"/v1/plans/{plan_id}",
                headers={"Authorization": "Api-Key test_api_key_12345"},
            )
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "invalid" in data["detail"].lower() or "format" in data["detail"].lower()

    def test_openapi_documentation_includes_endpoint(self) -> None:
        """Test that the endpoint is properly documented in the OpenAPI spec.

        Verifies that:
        - The endpoint appears in the OpenAPI schema
        - All response codes are documented (200, 401, 403, 404)
        - The endpoint has proper descriptions

        """
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        openapi_spec = response.json()
        assert "/v1/plans/{plan_id}" in openapi_spec["paths"]

        endpoint = openapi_spec["paths"]["/v1/plans/{plan_id}"]
        assert "get" in endpoint

        get_spec = endpoint["get"]
        assert "responses" in get_spec
        # Check that all required response codes are documented
        # Note: FastAPI may use string or int keys depending on version
        response_keys = {str(k) for k in get_spec["responses"].keys()}
        assert "200" in response_keys
        assert "401" in response_keys
        assert "403" in response_keys
        assert "404" in response_keys

        # Verify descriptions exist
        assert "description" in get_spec
        assert "summary" in get_spec

    def test_endpoint_requires_plan_id_parameter(self) -> None:
        """Test that the plan_id parameter is required.

        Verifies that:
        - Accessing /v1/plans/ without a plan_id returns 404 (route not found)

        """
        response = client.get(
            "/v1/plans/",
            headers={"Authorization": "Api-Key test_api_key_12345"},
        )

        # Should return 404 because the route requires {plan_id}
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
class TestAuthenticationFlow:
    """Test suite for authentication flow."""

    async def test_authentication_extracts_user_info(self) -> None:
        """Test that authentication correctly extracts user information from API key.

        This is a unit test for the authentication function itself.

        """
        from portia.api.auth import get_current_user

        # Test with valid API key format
        user = await get_current_user("Api-Key valid_key_12345")

        assert user is not None
        assert user.api_key == "valid_key_12345"
        assert user.user_id is not None
        assert user.organization_id is not None


@pytest.mark.asyncio
class TestAuthorizationChecks:
    """Test suite for authorization checks."""

    async def test_check_plan_access_allows_same_organization(self) -> None:
        """Test that users can access plans from their own organization."""
        from portia.api.auth import check_plan_access

        user = AuthenticatedUser(
            user_id="user_1",
            organization_id="org_123",
            api_key="test_key",
        )

        # Should not raise an exception
        check_plan_access("org_123", user)

    async def test_check_plan_access_denies_different_organization(self) -> None:
        """Test that users cannot access plans from other organizations."""
        from fastapi import HTTPException

        from portia.api.auth import check_plan_access

        user = AuthenticatedUser(
            user_id="user_1",
            organization_id="org_123",
            api_key="test_key",
        )

        # Should raise HTTPException with 403 status
        with pytest.raises(HTTPException) as exc_info:
            check_plan_access("org_999", user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
