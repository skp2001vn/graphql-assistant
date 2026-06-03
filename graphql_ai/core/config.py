from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class AppSettings:
    """Environment-backed application settings."""

    schema_file: Path = field(
        default_factory=lambda: Path(os.getenv("GRAPHQL_SCHEMA_FILE", "resources/schema.graphql"))
    )
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "graphql_schema"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "resources/models/all-MiniLM-L6-v2")
    )
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"))
    ollama_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")))
    ollama_num_predict: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_PREDICT", "1200")))
    ollama_think: bool = field(default_factory=lambda: _read_bool_env("OLLAMA_THINK"))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings for the current process."""
    return AppSettings()
