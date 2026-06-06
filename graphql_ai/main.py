from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphql_ai.agents import TroubleshootingAgent
from graphql_ai.api.routes import router
from graphql_ai.core.config import get_settings
from graphql_ai.core.responses import PrettyJSONResponse
from graphql_ai.llm.factory import build_llm_client
from graphql_ai.llm.pre_warm import LLMPreWarmer
from graphql_ai.rag.vector_store import SchemaVectorStore
from graphql_ai.services.sample_query_service import SampleQueryService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services and optionally pre-warm local Ollama inference."""
    settings = get_settings()
    schema_context_provider = SchemaVectorStore(settings=settings)
    llm_client = build_llm_client(settings)
    llm_pre_warmer = LLMPreWarmer(settings, llm_client)
    llm_pre_warmer.pre_warm()

    sample_service = SampleQueryService(
        settings=settings,
        llm_client=llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=schema_context_provider,
    )
    troubleshooting_agent = TroubleshootingAgent(
        settings=settings,
        llm_client=llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=schema_context_provider,
    )
    app.state.sample_service = sample_service
    app.state.troubleshooting_agent = troubleshooting_agent
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
