from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaChunk:
    """A chunk of GraphQL SDL prepared for retrieval."""

    id: str
    source: str
    kind: str
    name: str
    text: str


@dataclass(frozen=True)
class GeneratedGraphQLSample:
    """Generated GraphQL operation, variables, and raw model text."""

    operation: str
    variables: dict[str, Any]
    raw_response: str


@dataclass(frozen=True)
class TroubleshootingResult:
    """Troubleshooting result with validator issues and model-generated detail."""

    root_field: str
    status: str
    issues: list[str]
    detail: list[str]
    suggestion: str
    raw_response: str
