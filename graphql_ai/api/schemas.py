from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    """Natural-language request for the GraphQL AI assistant."""

    goal: str = Field(min_length=1, description="Natural-language assistant goal.")
    root_field: str = Field(
        min_length=1,
        description="GraphQL Query or Mutation root field the request focuses on.",
    )
    graphql_call: str | None = Field(
        default=None,
        description="GraphQL operation to troubleshoot when the assistant chooses troubleshooting.",
    )


class AssistantResultResponse(BaseModel):
    """HTTP response for the GraphQL AI assistant final result."""

    type: str
    operation: list[str] | None = None
    variables: dict[str, Any] | None = None
    root_field: str | None = None
    status: str | None = None
    issues: list[str] | None = None
    detail: list[str] | None = None
    suggestion: list[str] | None = None


class HealthResponse(BaseModel):
    """HTTP response for the health endpoint."""

    status: Literal["ok"]
