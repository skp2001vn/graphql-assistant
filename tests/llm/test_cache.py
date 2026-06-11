from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_assistant.llm.cache import CachedLLMClient, PromptResponseCache


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.pre_warm_calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"response for {prompt}"

    def pre_warm(self, prompt: str) -> None:
        self.pre_warm_calls.append(prompt)


class LLMCacheTest(unittest.TestCase):
    def test_prompt_response_cache_round_trips_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = PromptResponseCache(Path(temp_dir))

            cache.set("key", "response")

            self.assertEqual("response", cache.get("key"))

    def test_prompt_response_cache_ignores_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            (cache_dir / "bad.json").write_text("{", encoding="utf-8")
            cache = PromptResponseCache(cache_dir)

            self.assertIsNone(cache.get("bad"))

    def test_cached_llm_client_reuses_cached_response_for_same_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            llm_client = FakeLLMClient()
            cached_client = CachedLLMClient(
                llm_client=llm_client,
                cache=PromptResponseCache(Path(temp_dir)),
                namespace="test-model",
            )

            first_response = cached_client.generate("prompt")
            second_response = cached_client.generate("prompt")

        self.assertEqual("response for prompt", first_response)
        self.assertEqual(first_response, second_response)
        self.assertEqual(["prompt"], llm_client.calls)

    def test_cached_llm_client_namespaces_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = PromptResponseCache(Path(temp_dir))
            first_llm_client = FakeLLMClient()
            second_llm_client = FakeLLMClient()

            first_client = CachedLLMClient(first_llm_client, cache, "namespace-a")
            second_client = CachedLLMClient(second_llm_client, cache, "namespace-b")

            first_client.generate("same prompt")
            second_client.generate("same prompt")

        self.assertEqual(["same prompt"], first_llm_client.calls)
        self.assertEqual(["same prompt"], second_llm_client.calls)

    def test_cached_llm_client_pre_warm_bypasses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            llm_client = FakeLLMClient()
            cached_client = CachedLLMClient(
                llm_client=llm_client,
                cache=PromptResponseCache(Path(temp_dir)),
                namespace="test-model",
            )

            cached_client.generate("warm")
            cached_client.pre_warm("warm")

        self.assertEqual(["warm"], llm_client.calls)
        self.assertEqual(["warm"], llm_client.pre_warm_calls)


if __name__ == "__main__":
    unittest.main()
