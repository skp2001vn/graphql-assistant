from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for text-generation clients used by application services."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
