from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from graphql_assistant.agents import GraphQLAssistantAgent
from graphql_assistant.agents.tools import SampleTool, TroubleshootingTool
from graphql_assistant.api.routes import router
from graphql_assistant.core.config import get_settings
from graphql_assistant.core.responses import PrettyJSONResponse
from graphql_assistant.llm.factory import build_llm_client
from graphql_assistant.llm.pre_warm import LLMPreWarmer
from graphql_assistant.rag.vector_store import SchemaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize assistant tools and optionally pre-warm local Ollama inference."""
    settings = get_settings()
    schema_context_provider = SchemaVectorStore(settings=settings)
    llm_client = build_llm_client(settings)
    llm_pre_warmer = LLMPreWarmer(settings, llm_client)
    llm_pre_warmer.pre_warm()

    sample_tool = SampleTool(
        settings=settings,
        llm_client=llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=schema_context_provider,
    )
    troubleshooting_tool = TroubleshootingTool(
        settings=settings,
        llm_client=llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=schema_context_provider,
    )
    graphql_assistant_agent = GraphQLAssistantAgent(
        llm_client=llm_client,
        sample_tool=sample_tool,
        troubleshooting_tool=troubleshooting_tool,
    )
    app.state.graphql_assistant_agent = graphql_assistant_agent
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GraphQL Assistant Examples API",
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=PrettyJSONResponse,
    )
    app.include_router(router)
    return app


app = create_app()
