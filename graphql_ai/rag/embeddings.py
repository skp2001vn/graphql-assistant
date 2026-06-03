from __future__ import annotations

from pathlib import Path
from typing import Any


_embedding_model: tuple[str, Any] | None = None


def get_embedding_model(model_name_or_path: str, allow_downloads: bool = False) -> Any:
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
    embeddings = get_embedding_model(model_name_or_path, allow_downloads).encode(
        texts,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or "/" in value or "\\" in value

