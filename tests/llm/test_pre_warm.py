from __future__ import annotations

import unittest

from graphql_assistant.core.config import AppSettings
from graphql_assistant.llm.pre_warm import LLMPreWarmer


class FakeLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "ok"


class LLMPreWarmerTest(unittest.TestCase):
    def test_pre_warm_sends_configured_prompt_once_for_ollama(self) -> None:
        settings = AppSettings(
            inference_cache_enabled=False,
            ollama_pre_warm_enabled=True,
            ollama_pre_warm_prompt="warm",
        )
        llm_client = FakeLLMClient()
        pre_warmer = LLMPreWarmer(settings, llm_client)

        pre_warmer.pre_warm()
        pre_warmer.pre_warm()

        self.assertEqual(["warm"], llm_client.prompts)

    def test_pre_warm_skips_non_ollama_provider(self) -> None:
        settings = AppSettings(
            llm_provider="openai",
            inference_cache_enabled=False,
            ollama_pre_warm_enabled=True,
            ollama_pre_warm_prompt="warm",
        )
        llm_client = FakeLLMClient()
        pre_warmer = LLMPreWarmer(settings, llm_client)

        pre_warmer.pre_warm()

        self.assertEqual([], llm_client.prompts)


if __name__ == "__main__":
    unittest.main()
