from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaChunk:
    id: str
    source: str
    kind: str
    name: str
    text: str


@dataclass(frozen=True)
class GeneratedGraphQLSample:
    operation: str
    variables: dict[str, Any]
    raw_response: str

