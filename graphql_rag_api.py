from __future__ import annotations

import json
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from graphql_rag_local import build_vector_store, generate_graphql_sample, parse_generated_sample


class SampleQueryResponse(BaseModel):
    operation: list[str] = Field(description="Generated GraphQL operation formatted as lines.")
    variables: dict[str, Any] = Field(default_factory=dict, description="Variables for the operation.")


class HealthResponse(BaseModel):
    status: str


class PrettyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")


class GraphQLSampleService:
    def __init__(self) -> None:
        self.collection = build_vector_store()
        self._generation_lock = Lock()

    def generate(self, user_request: str):
        with self._generation_lock:
            raw_response = generate_graphql_sample(self.collection, user_request)

        return parse_generated_sample(raw_response)


def build_default_request(target: str) -> str:
    normalized_target = target.strip().replace("-", " ")
    if not normalized_target:
        raise ValueError("Target must not be empty.")

    if normalized_target.lower() == "country":
        return (
            "Generate a sample GraphQL query named CountryQuery for a country by code. "
            "Use a variable named code with type ID. "
            "Include all available Country fields, including continent and languages. "
            "Return Variables JSON with code set to US."
        )

    return f"Generate a sample query for {normalized_target}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sample_service = GraphQLSampleService()
    yield


app = FastAPI(
    title="GraphQL Local RAG API",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=PrettyJSONResponse,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get(
    "/generatesamplequery/{target}",
    response_model=SampleQueryResponse,
)
def generate_sample_query(
    target: str = Path(min_length=1, description="GraphQL resource or type, for example: country"),
    request: str | None = Query(default=None, description="Optional custom natural-language request."),
) -> SampleQueryResponse:
    user_request = request or build_default_request(target)
    sample_service: GraphQLSampleService = app.state.sample_service

    try:
        parsed_sample = sample_service.generate(user_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SampleQueryResponse(
        operation=parsed_sample.operation.splitlines(),
        variables=parsed_sample.variables,
    )
