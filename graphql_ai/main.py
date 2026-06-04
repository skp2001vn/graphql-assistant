from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphql_ai.api.routes import router
from graphql_ai.core.responses import PrettyJSONResponse
from graphql_ai.services.sample_query_service import SampleQueryService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services and pre-warm local inference for the API lifecycle."""
    sample_service = SampleQueryService()
    sample_service.pre_warm()
    app.state.sample_service = sample_service
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GraphQL AI Examples API",
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=PrettyJSONResponse,
    )
    app.include_router(router)
    return app


app = create_app()
