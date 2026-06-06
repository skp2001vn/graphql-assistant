from __future__ import annotations

import hashlib
import json
from pathlib import Path

from graphql_ai.llm.base import LLMClient


class PromptResponseCache:
    """File-backed inference cache for completed LLM prompt responses.

    Services cache final prompt/response pairs so repeated educational demos do
    not call the model provider again when the prompt and model settings match.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Create a cache rooted at the provided directory."""
        self.cache_dir = cache_dir

    def get(self, key: str) -> str | None:
        """Return a cached response for a key, if one exists.

        Invalid or partially written cache files are treated as cache misses so
        a bad local artifact does not break inference.
        """
        path = self._cache_path(key)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        response = payload.get("response")
        return response if isinstance(response, str) else None

    def set(self, key: str, response: str) -> None:
        """Store an LLM response under a deterministic cache key."""
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "response": response,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"


class CachedLLMClient:
    """LLM client wrapper that adds inference caching by prompt and namespace.

    The wrapper keeps caching separate from provider clients. The namespace
    captures model settings and workflow context, while the prompt text captures
    the exact request sent to inference.
    """

    def __init__(self, llm_client: LLMClient, cache: PromptResponseCache, namespace: str) -> None:
        """Create a cached LLM client around another LLM client."""
        self.llm_client = llm_client
        self.cache = cache
        self.namespace = namespace

    def generate(self, prompt: str) -> str:
        """Generate text, reusing a cached response for identical prompt input."""
        cache_key = self._cache_key(prompt)
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            print("Using cached LLM response.")
            return cached_response

        response = self.llm_client.generate(prompt)
        self.cache.set(cache_key, response)
        return response

    def pre_warm(self, prompt: str) -> None:
        """Pre-warm the wrapped provider without reading or writing inference cache."""
        pre_warm = getattr(self.llm_client, "pre_warm", None)
        if callable(pre_warm):
            pre_warm(prompt)
            return

        self.llm_client.generate(prompt)

    def _cache_key(self, prompt: str) -> str:
        key_material = json.dumps(
            {
                "namespace": self.namespace,
                "prompt": prompt,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()
