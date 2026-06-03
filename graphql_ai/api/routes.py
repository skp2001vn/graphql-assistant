from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, Request

from graphql_ai.api.schemas import HealthResponse, SampleQueryResponse
from graphql_ai.services.sample_query_service import SampleQueryService, build_default_sample_request


router = APIRouter()


def get_sample_service(request: Request) -> SampleQueryService:
    """Return the application-scoped sample-query service."""
    return request.app.state.sample_service


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health for local development and smoke tests."""
    return HealthResponse(status="ok")


@router.get("/sample/{target}", response_model=SampleQueryResponse)
def generate_sample_query(
    request: Request,
    target: str = Path(min_length=1, description="GraphQL resource or type, for example: country"),
    user_request: str | None = Query(
        default=None,
        alias="request",
        description="Optional custom natural-language request.",
    ),
) -> SampleQueryResponse:
    """Generate a sample GraphQL query for a schema target."""
    sample_request = user_request or build_default_sample_request(target)
    sample_service = get_sample_service(request)

    try:
        sample = sample_service.generate(sample_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SampleQueryResponse(
        operation=sample.operation.splitlines(),
        variables=sample.variables,
    )
