from __future__ import annotations

from typing import Protocol


class SchemaContextProvider(Protocol):
    """Protocol for components that provide schema context to assistant tools."""

    def retrieve_schema_context(self, retrieval_request: str) -> str:
        """Return schema context relevant to the retrieval request."""
