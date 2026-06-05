from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from graphql_ai.core.config import AppSettings, get_settings
from graphql_ai.rag.embeddings import embed_texts
from graphql_ai.rag.schema_chunks import SCHEMA_CHUNK_VERSION, load_schema_chunks


CACHE_METADATA_FILE = "index_metadata.json"
SCHEMA_CONTEXT_CACHE_VERSION = "5"


class SchemaVectorStore:
    """Chroma-backed RAG store for GraphQL schema chunks.

    This component is the main RAG implementation for the app. It owns the path
    from GraphQL SDL to prompt-ready schema context:

    1. Chunk the configured SDL file into definition-sized schema chunks.
    2. Embed those chunks with the configured embedding model.
    3. Persist chunk documents, embeddings, and metadata in Chroma.
    4. Embed a retrieval request such as "root field country".
    5. Query Chroma for the most relevant schema chunks.
    6. Format the retrieved chunks for prompt construction.

    The vector index is persisted in Chroma and invalidated when schema content,
    chunking version, embedding model, or collection name changes. Repeated
    request-to-context retrievals can also be cached separately so common demo
    calls avoid recomputing the request embedding and Chroma query.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        rebuild: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        """Build or load the schema vector index.

        `rebuild=True` forces a fresh Chroma collection. Otherwise, the store
        reuses a cached index when metadata proves it was built from the same
        schema content, chunking version, embedding model, and collection name.
        """
        self.settings = settings or get_settings()
        self.allow_downloads = allow_downloads
        self.collection = self._build_collection(rebuild=rebuild)
        self.schema_fingerprint = self._schema_fingerprint()

    def retrieve_schema_context(self, retrieval_request: str) -> str:
        """Retrieve prompt-ready schema context for a root-field request.

        This is the retrieval step services use before prompt construction. The
        method embeds the request, asks Chroma for the top-k closest schema
        chunks, and formats those chunks as text for the LLM prompt. When
        schema-context caching is enabled, the final formatted context is cached
        by request, schema fingerprint, top-k, and prompt-compression setting.
        """
        cache_key = self._schema_context_cache_key(retrieval_request)
        if self.settings.schema_context_cache_enabled:
            cached_context = self._read_schema_context_cache(cache_key)
            if cached_context is not None:
                print("Using cached schema context.")
                return cached_context

        collection_count = self.collection.count()
        n_results = min(self.settings.schema_context_top_k, collection_count)

        results = self.collection.query(
            query_embeddings=embed_texts(
                [retrieval_request],
                self.settings.embedding_model,
                self.allow_downloads,
            ),
            n_results=n_results,
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
        """Create the Chroma collection or reuse a valid cached index."""
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
        """Return metadata that identifies the schema index inputs.

        The fingerprint is used to decide whether the persisted Chroma index is
        still valid. It includes schema content, chunking version, embedding
        model, and collection name because any of those can change retrieval
        results.
        """
        schema_text = self.settings.schema_file.read_text(encoding="utf-8")
        return {
            "schema_file": str(self.settings.schema_file),
            "schema_sha1": hashlib.sha1(schema_text.encode("utf-8")).hexdigest(),
            "schema_chunk_version": SCHEMA_CHUNK_VERSION,
            "embedding_model": self.settings.embedding_model,
            "collection_name": self.settings.chroma_collection,
        }

    def _has_valid_cached_index(self) -> bool:
        """Return whether persisted Chroma metadata matches current settings."""
        return self._read_cache_metadata() == self._schema_fingerprint()

    def _cache_metadata_path(self) -> Path:
        """Return the path where Chroma index metadata is stored."""
        return Path(self.settings.chroma_path) / CACHE_METADATA_FILE

    def _read_cache_metadata(self) -> dict[str, Any] | None:
        """Read persisted Chroma index metadata, treating invalid JSON as absent."""
        path = self._cache_metadata_path()
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache_metadata(self, metadata: dict[str, Any]) -> None:
        """Persist metadata that describes the current Chroma index inputs."""
        path = self._cache_metadata_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _schema_context_cache_key(self, retrieval_request: str) -> str:
        """Build the cache key for formatted schema context.

        This cache is separate from Chroma's index cache. It stores the final
        text sent into prompt construction, so the key includes the retrieval
        request, schema fingerprint, top-k, prompt-compression setting, and an
        explicit cache version for formatting changes.
        """
        key_material = json.dumps(
            {
                "retrieval_request": retrieval_request,
                "schema_fingerprint": self.schema_fingerprint,
                "prompt_compression_enabled": self.settings.prompt_compression_enabled,
                "schema_context_top_k": self.settings.schema_context_top_k,
                "schema_context_cache_version": SCHEMA_CONTEXT_CACHE_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def _schema_context_cache_path(self, key: str) -> Path:
        """Return the file path for a formatted schema-context cache entry."""
        return self.settings.schema_context_cache_path / f"{key}.json"

    def _read_schema_context_cache(self, key: str) -> str | None:
        """Read a formatted schema-context cache entry if it is present."""
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
        """Write formatted schema context to the file-backed retrieval cache."""
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
        """Format a retrieved schema chunk for prompt construction.

        Prompt compression keeps each chunk on one compact line to reduce local
        inference latency. Disabling compression preserves the SDL block shape,
        which is useful when inspecting retrieval behavior during development.
        """
        if not self.settings.prompt_compression_enabled:
            return f"# {kind} {name}\n{document}"

        compact_document = " ".join(line.strip() for line in document.splitlines() if line.strip())
        return f"{kind} {name}: {compact_document}"
