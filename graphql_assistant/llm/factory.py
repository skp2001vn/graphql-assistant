from __future__ import annotations

from graphql_assistant.core.config import AppSettings
from graphql_assistant.llm.base import LLMClient
from graphql_assistant.llm.cache import CachedLLMClient, PromptResponseCache
from graphql_assistant.llm.ollama_client import OllamaClient
from graphql_assistant.llm.openai_client import OpenAIClient


def build_llm_client(settings: AppSettings, namespace_prefix: str = "") -> LLMClient:
    """Build the configured LLM provider client for assistant tools and agents.

    `LLM_PROVIDER` chooses the concrete adapter, currently `ollama` or
    `openai`. The default provider is local Ollama, which defaults to
    `OLLAMA_MODEL=qwen3:8b`. When inference caching is enabled, the provider is
    wrapped in `CachedLLMClient` with a namespace that includes model settings
    and an optional workflow prefix, such as `troubleshooting`.
    """
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
