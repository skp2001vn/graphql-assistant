from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from graphql_assistant.core.config import AppSettings
from graphql_assistant.llm.ollama_client import OllamaClient


class OllamaClientTest(unittest.TestCase):
    def test_generate_posts_expected_payload_and_returns_response_text(self) -> None:
        settings = AppSettings(
            ollama_url="http://ollama.test/api/generate",
            ollama_model="model",
            ollama_timeout_seconds=12,
            ollama_num_predict=100,
            ollama_num_ctx=2048,
            ollama_temperature=0,
            ollama_top_p=0.1,
            ollama_top_k=1,
            ollama_seed=42,
            ollama_keep_alive="20m",
            ollama_think=True,
        )
        response = Mock()
        response.json.return_value = {"response": " generated text "}
        response.raise_for_status.return_value = None

        with patch("requests.post", return_value=response) as post:
            result = OllamaClient(settings).generate("prompt")

        self.assertEqual("generated text", result)
        post.assert_called_once_with(
            "http://ollama.test/api/generate",
            json={
                "model": "model",
                "prompt": "prompt",
                "stream": False,
                "think": True,
                "keep_alive": "20m",
                "options": {
                    "num_predict": 100,
                    "temperature": 0,
                    "top_p": 0.1,
                    "top_k": 1,
                    "seed": 42,
                    "num_ctx": 2048,
                },
            },
            timeout=12,
        )

    def test_generate_excludes_num_ctx_when_not_configured(self) -> None:
        settings = AppSettings(ollama_num_ctx=None)
        response = Mock()
        response.json.return_value = {"response": "ok"}
        response.raise_for_status.return_value = None

        with patch("requests.post", return_value=response) as post:
            OllamaClient(settings).generate("prompt")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(
            {
                "num_predict": settings.ollama_num_predict,
                "temperature": settings.ollama_temperature,
                "top_p": settings.ollama_top_p,
                "top_k": settings.ollama_top_k,
                "seed": settings.ollama_seed,
            },
            payload["options"],
        )

    def test_generate_reports_missing_model_helpfully(self) -> None:
        settings = AppSettings(ollama_model="missing-model")
        response = Mock()
        response.status_code = 404
        response.text = "not found"

        import requests

        response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "ollama pull missing-model"):
                OllamaClient(settings).generate("prompt")


if __name__ == "__main__":
    unittest.main()
