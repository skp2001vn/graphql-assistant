from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request

from graphql_ai.agents import TroubleshootingAgent
from graphql_ai.api.schemas import HealthResponse, SampleQueryResponse, TroubleshootingResponse
from graphql_ai.services.sample_query_service import (
    InvalidRootFieldNameError,
    SampleQueryService,
)


router = APIRouter()


def get_sample_query_service(request: Request) -> SampleQueryService:
    """Return the application-scoped RAG and inference sample-query service."""
    return request.app.state.sample_service


def get_troubleshooting_agent(request: Request) -> TroubleshootingAgent:
    """Return the application-scoped GraphQL troubleshooting agent."""
    return request.app.state.troubleshooting_agent


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health for local development and smoke tests."""
    return HealthResponse(status="ok")


@router.get("/sample/{root_field}", response_model=SampleQueryResponse)
def generate_sample_query(
    request: Request,
    root_field: str = Path(
        min_length=1,
        description=(
            "GraphQL Query or Mutation field name to generate a sample for, "
            "for example: country"
        ),
    ),
) -> SampleQueryResponse:
    """Generate a sample GraphQL operation for a schema Query or Mutation field name."""
    try:
        sample_service = get_sample_query_service(request)
        sample = sample_service.generate(root_field)
    except InvalidRootFieldNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SampleQueryResponse(
        operation=sample.operation.splitlines(),
        variables=sample.variables,
    )


@router.post("/troubleshoot/{root_field}", response_model=TroubleshootingResponse)
async def troubleshoot_graphql_call(
    request: Request,
    root_field: str = Path(
        min_length=1,
        description=(
            "GraphQL Query or Mutation field name to troubleshoot, "
            "for example: country"
        ),
    ),
) -> TroubleshootingResponse:
    """Troubleshoot a submitted GraphQL call with an agent plan and local tools.

    The endpoint accepts either a plain-text GraphQL operation or Postman's
    GraphQL JSON body format: `{"query": "...", "variables": {...}}`.
    Variables are accepted for client compatibility; troubleshooting uses the
    query text.
    """
    try:
        graphql_call = await read_troubleshooting_graphql_call(request)
        troubleshooting_agent = get_troubleshooting_agent(request)
        result = troubleshooting_agent.troubleshoot(root_field, graphql_call)
    except InvalidRootFieldNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TroubleshootingResponse(
        root_field=result.root_field,
        status=result.status,
        issues=result.issues,
        detail=result.detail,
        suggestion=result.suggestion.splitlines() if result.suggestion else [],
    )


async def read_troubleshooting_graphql_call(request: Request) -> str:
    """Read a GraphQL operation from text/plain or Postman GraphQL JSON bodies."""
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Request JSON body must be valid JSON.") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
            raise HTTPException(
                status_code=400,
                detail="Request JSON body must include a string `query` field.",
            )

        return payload["query"]

    try:
        return (await request.body()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid UTF-8 text.") from exc
