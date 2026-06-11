from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_assistant.core.config import AppSettings
from graphql_assistant.llm.cache import CachedLLMClient
from graphql_assistant.llm.factory import build_llm_client
from graphql_assistant.llm.ollama_client import OllamaClient
from graphql_assistant.llm.openai_client import OpenAIClient


class LLMFactoryTest(unittest.TestCase):
    def test_builds_ollama_client_by_default(self) -> None:
        settings = AppSettings(inference_cache_enabled=False)

        llm_client = build_llm_client(settings)

        self.assertIsInstance(llm_client, OllamaClient)

    def test_builds_openai_client_when_provider_is_openai(self) -> None:
        settings = AppSettings(
            llm_provider="openai",
            inference_cache_enabled=False,
        )

        llm_client = build_llm_client(settings)

        self.assertIsInstance(llm_client, OpenAIClient)

    def test_wraps_provider_with_inference_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AppSettings(
                llm_provider="openai",
                openai_model="gpt-test",
                inference_cache_enabled=True,
                inference_cache_path=Path(temp_dir),
            )

            llm_client = build_llm_client(settings, namespace_prefix="troubleshooting")

        self.assertIsInstance(llm_client, CachedLLMClient)
        self.assertIsInstance(llm_client.llm_client, OpenAIClient)
        self.assertEqual(
            f"troubleshooting|{settings.inference_cache_namespace()}",
            llm_client.namespace,
        )

    def test_rejects_unknown_provider(self) -> None:
        settings = AppSettings(llm_provider="unknown")

        with self.assertRaisesRegex(RuntimeError, "Unsupported LLM_PROVIDER: unknown"):
            build_llm_client(settings)


if __name__ == "__main__":
    unittest.main()
