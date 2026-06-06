from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaChunk:
    """GraphQL SDL fragment prepared for embedding and retrieval.

    RAG stores these chunks in the vector store, then retrieval returns the
    most relevant chunks as schema context for prompt construction.
    """

    id: str
    source: str
    kind: str
    name: str
    text: str


@dataclass(frozen=True)
class GeneratedGraphQLSample:
    """Generated sample operation returned by the sample-query workflow.

    `operation` is the GraphQL text after parsing model output, `variables`
    is the Variables JSON object paired with that operation, and
    `raw_response` preserves the original model text for troubleshooting.
    """

    operation: str
    variables: dict[str, Any]
    raw_response: str


@dataclass(frozen=True)
class TroubleshootingResult:
    """Result returned by the troubleshooting service workflow.

    The result keeps deterministic GraphQL validation issues separate from
    model-generated detail text and the validated suggested operation.
    """

    root_field: str
    status: str
    issues: list[str]
    detail: list[str]
    suggestion: str
    raw_response: str
