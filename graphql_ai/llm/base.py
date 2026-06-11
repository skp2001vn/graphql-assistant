from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for text-generation clients used by assistant tools and agents."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class WarmableLLMClient(LLMClient, Protocol):
    """Protocol for provider clients that can pre-load model resources."""

    def pre_warm(self, prompt: str) -> None:
        """Pre-load provider resources before custom inference."""
