from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SampleQueryResponse(BaseModel):
    """HTTP response for generated sample GraphQL operations."""

    operation: list[str] = Field(description="Generated GraphQL operation formatted as lines.")
    variables: dict[str, Any] = Field(default_factory=dict, description="Variables for the operation.")


class TroubleshootingResponse(BaseModel):
    """HTTP response for the GraphQL troubleshooting agent."""

    root_field: str = Field(description="GraphQL Query or Mutation field name being troubleshot.")
    status: str = Field(description="Validation status for the submitted GraphQL operation.")
    issues: list[str] = Field(default_factory=list, description="Syntax or schema issues found by tools.")
    detail: str = Field(description="Agent-generated troubleshooting guidance.")
    suggestion: list[str] = Field(
        default_factory=list,
        description="Agent-suggested GraphQL operation formatted as lines.",
    )


class HealthResponse(BaseModel):
    """HTTP response for the health endpoint."""

    status: str
