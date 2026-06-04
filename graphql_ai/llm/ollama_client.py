from __future__ import annotations

from graphql_ai.core.config import AppSettings, get_settings


class OllamaClient:
    """Small HTTP client for local Ollama inference."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Create an Ollama client from application settings."""
        self.settings = settings or get_settings()

    def generate(self, prompt: str) -> str:
        """Generate text with runtime options tuned for local inference.

        `keep_alive` keeps the model loaded between requests, `num_predict`
        limits maximum output length, `num_ctx` can constrain or expand the
        context window, and `think` controls model reasoning mode when the local
        model supports it.
        """
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install requests with `pip install -r requirements.txt`.") from exc

        options = {
            "num_predict": self.settings.ollama_num_predict,
        }
        if self.settings.ollama_num_ctx is not None:
            options["num_ctx"] = self.settings.ollama_num_ctx

        response = requests.post(
            self.settings.ollama_url,
            json={
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": self.settings.ollama_think,
                "keep_alive": self.settings.ollama_keep_alive,
                "options": options,
            },
            timeout=self.settings.ollama_timeout_seconds,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama model or endpoint not found.\n"
                    f"Configured model: {self.settings.ollama_model}\n"
                    f"Pull the model first with:\n"
                    f"  ollama pull {self.settings.ollama_model}\n"
                    f"Ollama response: {response.text}"
                ) from exc

            raise RuntimeError(f"Ollama request failed: {response.text}") from exc

        return str(response.json().get("response", "")).strip()
