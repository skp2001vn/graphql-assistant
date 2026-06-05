from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from graphql_ai.core.config import AppSettings, get_settings


class ConfigTest(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_app_settings_reads_environment_values(self) -> None:
        env = {
            "GRAPHQL_SCHEMA_FILE": "custom/schema.graphql",
            "SCHEMA_CONTEXT_TOP_K": "3",
            "LLM_PROVIDER": "openai",
            "OLLAMA_MODEL": "test-model",
            "OLLAMA_NUM_PREDICT": "123",
            "OLLAMA_NUM_CTX": "2048",
            "OLLAMA_TEMPERATURE": "0.2",
            "OLLAMA_TOP_P": "0.3",
            "OLLAMA_TOP_K": "4",
            "OLLAMA_SEED": "99",
            "OLLAMA_KEEP_ALIVE": "30m",
            "OLLAMA_THINK": "true",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_URL": "https://openai.test/v1/responses",
            "OPENAI_MODEL": "gpt-test",
            "OPENAI_TIMEOUT_SECONDS": "45",
            "OPENAI_MAX_OUTPUT_TOKENS": "321",
            "INFERENCE_CACHE_ENABLED": "false",
            "PROMPT_COMPRESSION_ENABLED": "false",
            "PROMPT_CONTRACT_VERSION": "test-contract",
            "OLLAMA_PRE_WARM_ENABLED": "false",
        }

        with patch.dict(os.environ, env, clear=False):
            settings = AppSettings()

        self.assertEqual("custom/schema.graphql", str(settings.schema_file))
        self.assertEqual(3, settings.schema_context_top_k)
        self.assertEqual("openai", settings.llm_provider)
        self.assertEqual("test-model", settings.ollama_model)
        self.assertEqual(123, settings.ollama_num_predict)
        self.assertEqual(2048, settings.ollama_num_ctx)
        self.assertEqual(0.2, settings.ollama_temperature)
        self.assertEqual(0.3, settings.ollama_top_p)
        self.assertEqual(4, settings.ollama_top_k)
        self.assertEqual(99, settings.ollama_seed)
        self.assertEqual("30m", settings.ollama_keep_alive)
        self.assertTrue(settings.ollama_think)
        self.assertEqual("test-key", settings.openai_api_key)
        self.assertEqual("https://openai.test/v1/responses", settings.openai_url)
        self.assertEqual("gpt-test", settings.openai_model)
        self.assertEqual(45, settings.openai_timeout_seconds)
        self.assertEqual(321, settings.openai_max_output_tokens)
        self.assertFalse(settings.inference_cache_enabled)
        self.assertFalse(settings.prompt_compression_enabled)
        self.assertEqual("test-contract", settings.prompt_contract_version)
        self.assertFalse(settings.ollama_pre_warm_enabled)

    def test_inference_cache_namespace_includes_generation_settings(self) -> None:
        settings = AppSettings(
            ollama_model="model",
            ollama_num_predict=10,
            ollama_num_ctx=20,
            ollama_temperature=0,
            ollama_top_p=0.1,
            ollama_top_k=1,
            ollama_seed=42,
            ollama_keep_alive="5m",
            ollama_think=True,
            prompt_compression_enabled=False,
            prompt_contract_version="contract",
            ollama_pre_warm_enabled=False,
        )

        namespace = settings.inference_cache_namespace()

        self.assertEqual("ollama|model|10|20|0|0.1|1|42|5m|True|False|False|contract", namespace)

    def test_openai_inference_cache_namespace_includes_provider_settings(self) -> None:
        settings = AppSettings(
            llm_provider="openai",
            openai_model="gpt-test",
            openai_max_output_tokens=123,
            prompt_compression_enabled=False,
            prompt_contract_version="contract",
        )

        namespace = settings.inference_cache_namespace()

        self.assertEqual("openai|gpt-test|123|False|contract", namespace)

    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()

        first = get_settings()
        second = get_settings()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
