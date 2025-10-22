"""Main FastAPI application for the Portia Cloud API.

This module sets up the FastAPI application with all routes, middleware,
and configuration needed to run the Portia Cloud backend API.

"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from portia.api.routes import router

# Create the FastAPI application
app = FastAPI(
    title="Portia Cloud API",
    description="""
    Backend API for Portia Cloud - an AI agent workflow platform.

    This API provides endpoints for managing plans, runs, and other resources
    in the Portia Cloud ecosystem. All endpoints require authentication via
    API key and enforce organization-based access control.

    ## Authentication

    All API endpoints require authentication using an API key in the Authorization header:

    ```
    Authorization: Api-Key <your-api-key>
    ```

    ## Authorization

    Users can only access resources (plans, runs, etc.) that belong to their organization.
    Attempting to access resources from other organizations will result in a 403 Forbidden error.

    ## Rate Limiting

    API requests are rate-limited to ensure fair usage. Contact support if you need
    higher limits for your use case.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the plans router
app.include_router(router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Root endpoint for health checks.

    Returns:
        dict: A simple message indicating the API is running.

    """
    return {
        "message": "Portia Cloud API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict: Health status of the API.

    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # Run the application
    # In production, use a proper ASGI server like gunicorn with uvicorn workers
    uvicorn.run(app, host="0.0.0.0", port=8000)
