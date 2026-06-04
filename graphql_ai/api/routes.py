from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request

from graphql_ai.api.schemas import HealthResponse, SampleQueryResponse
from graphql_ai.services.sample_query_service import SampleQueryService


router = APIRouter()


def get_sample_query_service(request: Request) -> SampleQueryService:
    """Return the application-scoped RAG and inference sample-query service."""
    return request.app.state.sample_service


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
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SampleQueryResponse(
        operation=sample.operation.splitlines(),
        variables=sample.variables,
    )
