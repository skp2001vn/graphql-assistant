from __future__ import annotations

from graphql_ai.core.config import AppSettings
from graphql_ai.llm.base import LLMClient
from graphql_ai.llm.cache import CachedLLMClient, PromptResponseCache
from graphql_ai.llm.ollama_client import OllamaClient
from graphql_ai.llm.openai_client import OpenAIClient


def build_llm_client(settings: AppSettings, namespace_prefix: str = "") -> LLMClient:
    """Build the configured LLM provider client, wrapped with inference caching when enabled."""
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        llm_client: LLMClient = OllamaClient(settings=settings)
    elif provider == "openai":
        llm_client = OpenAIClient(settings=settings)
    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    if not settings.inference_cache_enabled:
        return llm_client

    namespace = settings.inference_cache_namespace()
    if namespace_prefix:
        namespace = f"{namespace_prefix}|{namespace}"

    return CachedLLMClient(
        llm_client=llm_client,
        cache=PromptResponseCache(settings.inference_cache_path),
        namespace=namespace,
    )
