from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from graphql_assistant.core.config import AppSettings
from graphql_assistant.llm.openai_client import OpenAIClient


class OpenAIClientTest(unittest.TestCase):
    def test_generate_posts_expected_payload_and_returns_output_text(self) -> None:
        settings = AppSettings(
            openai_api_key="test-key",
            openai_url="https://openai.test/v1/responses",
            openai_model="gpt-test",
            openai_timeout_seconds=12,
            openai_max_output_tokens=345,
        )
        response = Mock()
        response.json.return_value = {"output_text": " generated text "}
        response.raise_for_status.return_value = None

        with patch("requests.post", return_value=response) as post:
            result = OpenAIClient(settings).generate("prompt")

        self.assertEqual("generated text", result)
        post.assert_called_once_with(
            "https://openai.test/v1/responses",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-test",
                "input": "prompt",
                "max_output_tokens": 345,
            },
            timeout=12,
        )

    def test_generate_reads_output_message_content(self) -> None:
        settings = AppSettings(openai_api_key="test-key")
        response = Mock()
        response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "part one"},
                        {"type": "output_text", "text": " part two"},
                    ],
                }
            ]
        }
        response.raise_for_status.return_value = None

        with patch("requests.post", return_value=response):
            result = OpenAIClient(settings).generate("prompt")

        self.assertEqual("part one part two", result)

    def test_generate_requires_api_key(self) -> None:
        settings = AppSettings(openai_api_key=None)

        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            OpenAIClient(settings).generate("prompt")

    def test_generate_reports_api_errors(self) -> None:
        settings = AppSettings(openai_api_key="test-key")
        response = Mock()
        response.text = "bad request"

        import requests

        response.raise_for_status.side_effect = requests.HTTPError("400")

        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "OpenAI request failed: bad request"):
                OpenAIClient(settings).generate("prompt")

    def test_generate_reports_missing_output_text(self) -> None:
        settings = AppSettings(openai_api_key="test-key")
        response = Mock()
        response.json.return_value = {"output": []}
        response.raise_for_status.return_value = None

        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "did not include output text"):
                OpenAIClient(settings).generate("prompt")


if __name__ == "__main__":
    unittest.main()
