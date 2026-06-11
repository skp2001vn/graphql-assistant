from __future__ import annotations

from typing import Any

from graphql_ai.core.config import AppSettings, get_settings


class OpenAIClient:
    """Small HTTP client for OpenAI Responses API inference."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Create an OpenAI client from application settings."""
        self.settings = settings or get_settings()

    def generate(self, prompt: str) -> str:
        """Generate text with OpenAI using the configured Responses API model.

        The application passes a single prompt string because prompt
        construction already happened in the assistant tool or agent layer. This
        adapter only owns provider-specific HTTP details: API key validation,
        request payload shape, timeout, error translation, and output-text
        extraction.
        """
        if not self.settings.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for LLM_PROVIDER=openai.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install requests with `pip install -r requirements.txt`.") from exc

        response = requests.post(
            self.settings.openai_url,
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.openai_model,
                "input": prompt,
                "max_output_tokens": self.settings.openai_max_output_tokens,
            },
            timeout=self.settings.openai_timeout_seconds,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"OpenAI request failed: {response.text}") from exc

        output_text = _extract_response_text(response.json())
        if not output_text:
            raise RuntimeError("OpenAI response did not include output text.")

        return output_text.strip()


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    text_parts = []
    output_items = payload.get("output")
    if not isinstance(output_items, list):
        return ""

    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue
        content_items = output_item.get("content")
        if not isinstance(content_items, list):
            continue
        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                text_parts.append(content_item["text"])

    return "".join(text_parts)
