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
    def __init__(
        self,
        settings: AppSettings | None = None,
        rebuild: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.allow_downloads = allow_downloads
        self.collection = self._build_collection(rebuild=rebuild)

    def retrieve_schema_context(self, user_request: str) -> str:
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
            source = metadata.get("source", "unknown")
            kind = metadata.get("kind", "definition")
            name = metadata.get("name", "unknown")
            context_parts.append(f"# Source: {source} ({kind} {name})\n{document}")

        return "\n\n".join(context_parts)

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

