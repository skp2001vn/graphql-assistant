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
            "OLLAMA_MODEL": "test-model",
            "OLLAMA_NUM_PREDICT": "123",
            "OLLAMA_NUM_CTX": "2048",
            "OLLAMA_KEEP_ALIVE": "30m",
            "OLLAMA_THINK": "true",
            "INFERENCE_CACHE_ENABLED": "false",
            "PROMPT_COMPRESSION_ENABLED": "false",
            "OLLAMA_PRE_WARM_ENABLED": "false",
        }

        with patch.dict(os.environ, env, clear=False):
            settings = AppSettings()

        self.assertEqual("custom/schema.graphql", str(settings.schema_file))
        self.assertEqual("test-model", settings.ollama_model)
        self.assertEqual(123, settings.ollama_num_predict)
        self.assertEqual(2048, settings.ollama_num_ctx)
        self.assertEqual("30m", settings.ollama_keep_alive)
        self.assertTrue(settings.ollama_think)
        self.assertFalse(settings.inference_cache_enabled)
        self.assertFalse(settings.prompt_compression_enabled)
        self.assertFalse(settings.ollama_pre_warm_enabled)

    def test_inference_cache_namespace_includes_generation_settings(self) -> None:
        settings = AppSettings(
            ollama_model="model",
            ollama_num_predict=10,
            ollama_num_ctx=20,
            ollama_keep_alive="5m",
            ollama_think=True,
            prompt_compression_enabled=False,
            ollama_pre_warm_enabled=False,
        )

        namespace = settings.inference_cache_namespace()

        self.assertEqual("model|10|20|5m|True|False|False", namespace)

    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()

        first = get_settings()
        second = get_settings()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
