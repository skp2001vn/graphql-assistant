from __future__ import annotations

from threading import Lock

from graphql_ai.core.config import AppSettings
from graphql_ai.llm.base import LLMClient


class LLMPreWarmer:
    """Coordinates model pre-warm for configured LLM provider clients.

    Model pre-warm is an inference optimization for local providers such as
    Ollama. It belongs in the LLM layer so assistant tools and agents can share the
    same warm-once behavior without owning provider-specific checks.
    """

    def __init__(self, settings: AppSettings, llm_client: LLMClient) -> None:
        """Create a pre-warmer for a configured LLM client."""
        self.settings = settings
        self.llm_client = llm_client
        self._lock = Lock()
        self._pre_warmed = False

    def pre_warm(self) -> None:
        """Pre-load provider resources once before custom inference."""
        with self._lock:
            if self._pre_warmed:
                return

            if self.settings.llm_provider.lower() != "ollama" or not self.settings.ollama_pre_warm_enabled:
                self._pre_warmed = True
                return

            pre_warm = getattr(self.llm_client, "pre_warm", None)
            if callable(pre_warm):
                pre_warm(self.settings.ollama_pre_warm_prompt)
            else:
                self.llm_client.generate(self.settings.ollama_pre_warm_prompt)

            self._pre_warmed = True
