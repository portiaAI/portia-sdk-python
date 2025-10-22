# Portia Cloud Backend API

This module contains the backend API implementation for Portia Cloud, providing RESTful endpoints for managing plans, runs, and other resources.

## Overview

The API is built using FastAPI and provides:
- **Authentication**: API key-based authentication via the `Authorization` header
- **Authorization**: Organization-based access control
- **OpenAPI Documentation**: Auto-generated API documentation at `/docs` and `/redoc`
- **Error Handling**: Consistent error responses for 400, 401, 403, 404, and 500 errors

## Installation

Install the API dependencies:

```bash
pip install portia-sdk-python[api]
```

Or install all optional dependencies including API:

```bash
pip install portia-sdk-python[all]
```

## Running the Server

### Development

Run the server in development mode:

```bash
python -m portia.api.app
```

Or using uvicorn directly:

```bash
uvicorn portia.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Production

For production deployments, use gunicorn with uvicorn workers:

```bash
gunicorn portia.api.app:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

## API Endpoints

### GET /v1/plans/{plan_id}

Retrieve a single plan by its unique ID.

**Authentication**: Required (API Key)

**Authorization**: User must belong to the same organization as the plan

**Request**:
```bash
curl -X GET "http://localhost:8000/v1/plans/plan_abc123" \
  -H "Authorization: Api-Key YOUR_API_KEY"
```

**Success Response** (200 OK):
```json
{
  "id": "plan_abc123def456",
  "query": "Find the best pizza places in New York",
  "tool_ids": ["portia:web_search", "portia:maps"],
  "steps": [
    {
      "task": "Search for pizza places in New York",
      "tool_id": "portia:web_search",
      "output": "$search_results",
      "inputs": []
    },
    {
      "task": "Find the highest rated option",
      "tool_id": null,
      "output": "$best_pizza_place",
      "inputs": []
    }
  ],
  "plan_inputs": []
}
```

**Error Responses**:
- **400 Bad Request**: Invalid plan ID format
- **401 Unauthorized**: Missing or invalid API key
- **403 Forbidden**: User does not have permission to access this plan
- **404 Not Found**: Plan does not exist

## Authentication

All API endpoints require authentication using an API key in the `Authorization` header:

```
Authorization: Api-Key YOUR_API_KEY
```

## Authorization

Users can only access resources (plans, runs, etc.) that belong to their organization. The API enforces this by checking the organization ID associated with the authenticated user against the organization that owns the requested resource.

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Database Integration

The current implementation provides the interface for database access in `portia/api/database.py`. To integrate with a real database:

1. Update the `DatabaseClient` class to connect to your database
2. Implement the `get_plan_by_id` method to query your database
3. Update the authentication logic in `portia/api/auth.py` to validate API keys against your database

Example PostgreSQL integration:

```python
import asyncpg
from portia.plan import Plan, PlanUUID

class DatabaseClient:
    def __init__(self, connection_string: str):
        self.pool = await asyncpg.create_pool(connection_string)

    async def get_plan_by_id(self, plan_id: PlanUUID) -> tuple[Plan, str]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM plans WHERE id = $1",
                str(plan_id)
            )
            if not row:
                raise PlanNotFoundError(plan_id)

            plan = Plan.from_response(dict(row))
            organization_id = row['organization_id']
            return plan, organization_id
```

## Testing

Run the tests:

```bash
pytest tests/unit/api/
```

Run tests with coverage:

```bash
pytest tests/unit/api/ --cov=portia/api --cov-report=html
```

## Architecture

The API follows a layered architecture:

1. **Routes** (`routes.py`): Define FastAPI endpoints and request/response handling
2. **Authentication** (`auth.py`): Handle API key validation and user authentication
3. **Authorization** (`auth.py`): Enforce organization-based access control
4. **Database** (`database.py`): Abstract database access layer
5. **Models** (`models.py`): Pydantic models for request/response validation

## Security Considerations

1. **API Keys**: In production, store API keys securely (hashed) in the database
2. **HTTPS**: Always use HTTPS in production to protect API keys in transit
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Input Validation**: All inputs are validated using Pydantic models
5. **CORS**: Configure CORS appropriately for your frontend domains

## Future Enhancements

- [ ] Add pagination support for list endpoints
- [ ] Implement rate limiting
- [ ] Add request ID tracking for debugging
- [ ] Add metrics and monitoring (Prometheus, etc.)
- [ ] Implement caching layer (Redis)
- [ ] Add webhook support for async operations
- [ ] Support multiple authentication methods (OAuth2, JWT)
