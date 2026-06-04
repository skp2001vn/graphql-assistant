from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from graphql_ai.core.config import AppSettings, get_settings
from graphql_ai.rag.embeddings import embed_texts
from graphql_ai.rag.schema_chunks import load_schema_chunks


CACHE_METADATA_FILE = "index_metadata.json"


class SchemaVectorStore:
    """Chroma-backed RAG store for GraphQL schema chunks.

    This component owns schema chunking output, embeddings, vector-store
    persistence, retrieval, schema-context formatting, and retrieval caching.
    The vector index is persisted in Chroma, while repeated request-to-context
    retrievals can be cached separately. This avoids recomputing the request
    embedding and Chroma query when the same prompt is repeated.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        rebuild: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        """Build or load the cached schema vector index."""
        self.settings = settings or get_settings()
        self.allow_downloads = allow_downloads
        self.collection = self._build_collection(rebuild=rebuild)
        self.schema_fingerprint = self._schema_fingerprint()

    def retrieve_schema_context(self, user_request: str) -> str:
        """Retrieve compact schema chunks relevant to a natural-language request.

        This is the RAG retrieval step used before prompt construction. When
        enabled, this method caches the final schema context by request, schema
        fingerprint, embedding model, collection name, and prompt compression
        setting.
        """
        cache_key = self._schema_context_cache_key(user_request)
        if self.settings.schema_context_cache_enabled:
            cached_context = self._read_schema_context_cache(cache_key)
            if cached_context is not None:
                print("Using cached schema context.")
                return cached_context

        results = self.collection.query(
            query_embeddings=embed_texts(
                [user_request],
                self.settings.embedding_model,
                self.allow_downloads,
            ),
            n_results=self.collection.count(),
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        context_parts = []
        for document, metadata in zip(documents, metadatas):
            kind = metadata.get("kind", "definition")
            name = metadata.get("name", "unknown")
            chunk_text = self._format_schema_chunk(document, kind, name)
            context_parts.append(chunk_text)

        schema_context = "\n\n".join(context_parts)
        if self.settings.schema_context_cache_enabled:
            self._write_schema_context_cache(cache_key, schema_context)

        return schema_context

    def _build_collection(self, rebuild: bool):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install chromadb with `pip install -r requirements.txt`."
            ) from exc

        client = chromadb.PersistentClient(
            path=self.settings.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        if not rebuild and self._has_valid_cached_index():
            collection = client.get_or_create_collection(self.settings.chroma_collection)
            if collection.count() > 0:
                print(f"Using cached GraphQL schema index from {self.settings.chroma_path}.")
                return collection

            print("Cached schema index metadata exists, but the Chroma collection is empty; rebuilding.")
        elif not rebuild:
            print("Schema index cache is missing or stale; rebuilding.")

        try:
            client.delete_collection(self.settings.chroma_collection)
        except Exception:
            pass

        collection = client.get_or_create_collection(self.settings.chroma_collection)
        chunks = load_schema_chunks(self.settings.schema_file)
        documents = [chunk.text for chunk in chunks]

        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=documents,
            embeddings=embed_texts(documents, self.settings.embedding_model, self.allow_downloads),
            metadatas=[
                {
                    "source": chunk.source,
                    "kind": chunk.kind,
                    "name": chunk.name,
                }
                for chunk in chunks
            ],
        )

        self._write_cache_metadata(self._schema_fingerprint())
        print(f"Indexed {len(chunks)} GraphQL schema chunks from {self.settings.schema_file}.")
        return collection

    def _schema_fingerprint(self) -> dict[str, Any]:
        schema_text = self.settings.schema_file.read_text(encoding="utf-8")
        return {
            "schema_file": str(self.settings.schema_file),
            "schema_sha1": hashlib.sha1(schema_text.encode("utf-8")).hexdigest(),
            "embedding_model": self.settings.embedding_model,
            "collection_name": self.settings.chroma_collection,
        }

    def _has_valid_cached_index(self) -> bool:
        return self._read_cache_metadata() == self._schema_fingerprint()

    def _cache_metadata_path(self) -> Path:
        return Path(self.settings.chroma_path) / CACHE_METADATA_FILE

    def _read_cache_metadata(self) -> dict[str, Any] | None:
        path = self._cache_metadata_path()
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache_metadata(self, metadata: dict[str, Any]) -> None:
        path = self._cache_metadata_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _schema_context_cache_key(self, user_request: str) -> str:
        key_material = json.dumps(
            {
                "user_request": user_request,
                "schema_fingerprint": self.schema_fingerprint,
                "prompt_compression_enabled": self.settings.prompt_compression_enabled,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def _schema_context_cache_path(self, key: str) -> Path:
        return self.settings.schema_context_cache_path / f"{key}.json"

    def _read_schema_context_cache(self, key: str) -> str | None:
        path = self._schema_context_cache_path(key)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        schema_context = payload.get("schema_context")
        return schema_context if isinstance(schema_context, str) else None

    def _write_schema_context_cache(self, key: str, schema_context: str) -> None:
        path = self._schema_context_cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "schema_context": schema_context,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _format_schema_chunk(self, document: str, kind: str, name: str) -> str:
        if not self.settings.prompt_compression_enabled:
            return f"# {kind} {name}\n{document}"

        compact_document = " ".join(line.strip() for line in document.splitlines() if line.strip())
        return f"{kind} {name}: {compact_document}"
