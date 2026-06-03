from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SampleQueryResponse(BaseModel):
    operation: list[str] = Field(description="Generated GraphQL operation formatted as lines.")
    variables: dict[str, Any] = Field(default_factory=dict, description="Variables for the operation.")


class HealthResponse(BaseModel):
    status: str

