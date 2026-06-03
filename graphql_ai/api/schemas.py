from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SampleQueryResponse(BaseModel):
    """HTTP response for generated sample GraphQL operations."""

    operation: list[str] = Field(description="Generated GraphQL operation formatted as lines.")
    variables: dict[str, Any] = Field(default_factory=dict, description="Variables for the operation.")


class HealthResponse(BaseModel):
    """HTTP response for the health endpoint."""

    status: str
