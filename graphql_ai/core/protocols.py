from __future__ import annotations

from typing import Protocol


class SchemaContextProvider(Protocol):
    def retrieve_schema_context(self, user_request: str) -> str:
        """Return schema context relevant to the user's request."""

