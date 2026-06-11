from __future__ import annotations

from pathlib import Path
from typing import Any


_embedding_model: tuple[str, Any] | None = None


def get_embedding_model(model_name_or_path: str, allow_downloads: bool = False) -> Any:
    """Load and cache the sentence-transformers embedding model.

    RAG starts by turning schema chunks and retrieval requests into embedding
    vectors. Loading the embedding model is relatively expensive, so this module
    keeps one model instance in memory and reuses it until the configured model
    name or local path changes.

    By default, this application expects the embedding model to already
    exist under `resources/models/`. `allow_downloads=True` is reserved for
    setup or live integration paths that are allowed to reach the network.
    """
    global _embedding_model

    if _embedding_model is None or _embedding_model[0] != model_name_or_path:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install sentence-transformers with "
                "`pip install -r requirements.txt`."
            ) from exc

        model_path = Path(model_name_or_path)
        if not allow_downloads and _looks_like_local_path(model_name_or_path) and not model_path.exists():
            raise RuntimeError(
                f"Local embedding model not found: {model_path}\n"
                "Download it once while online, then run again locally:\n"
                "  python -c \"from sentence_transformers import SentenceTransformer; "
                "SentenceTransformer('all-MiniLM-L6-v2').save_pretrained("
                "'resources/models/all-MiniLM-L6-v2')\""
            )

        _embedding_model = (
            model_name_or_path,
            SentenceTransformer(model_name_or_path, local_files_only=not allow_downloads),
        )

    return _embedding_model[1]


def embed_texts(
    texts: list[str],
    model_name_or_path: str,
    allow_downloads: bool = False,
) -> list[list[float]]:
    """Convert text values into normalized embedding vectors for retrieval.

    Chroma compares the request embedding with stored schema-chunk embeddings.
    Normalizing embeddings makes similarity search more stable across chunks of
    different lengths and keeps the vector-store behavior stable for repeated
    local runs.
    """
    embeddings = get_embedding_model(model_name_or_path, allow_downloads).encode(
        texts,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or "/" in value or "\\" in value
