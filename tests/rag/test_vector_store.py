from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graphql_ai.core.config import AppSettings
from graphql_ai.rag.vector_store import SchemaVectorStore


SCHEMA = """
type Query {
  country(code: ID!): Country
}

type Country {
  code: ID!
}
"""


class FakeCollection:
    def __init__(self) -> None:
        self.query_calls = 0
        self.n_results: list[int] = []

    def count(self) -> int:
        return 10

    def query(self, query_embeddings: list[list[float]], n_results: int) -> dict[str, list[list[object]]]:
        self.query_calls += 1
        self.n_results.append(n_results)
        return {
            "documents": [["type Country {\n  code: ID!\n}"]],
            "metadatas": [[{"kind": "type", "name": "Country"}]],
        }


class VectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.schema_file = root / "schema.graphql"
        self.schema_file.write_text(SCHEMA, encoding="utf-8")
        self.settings = AppSettings(
            schema_file=self.schema_file,
            chroma_path=str(root / "chroma"),
            schema_context_cache_path=root / "schema_context",
            embedding_model="test-embedding-model",
            prompt_compression_enabled=True,
            schema_context_top_k=4,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build_store_without_chroma(self) -> SchemaVectorStore:
        store = SchemaVectorStore.__new__(SchemaVectorStore)
        store.settings = self.settings
        store.allow_downloads = False
        store.collection = FakeCollection()
        store.schema_fingerprint = store._schema_fingerprint()
        return store

    def test_schema_fingerprint_changes_when_schema_changes(self) -> None:
        store = self.build_store_without_chroma()
        first_fingerprint = store._schema_fingerprint()

        self.schema_file.write_text(f"{SCHEMA}\nscalar DateTime\n", encoding="utf-8")
        second_fingerprint = store._schema_fingerprint()

        self.assertNotEqual(first_fingerprint["schema_sha1"], second_fingerprint["schema_sha1"])

    def test_cache_metadata_round_trips(self) -> None:
        store = self.build_store_without_chroma()
        metadata = store._schema_fingerprint()

        store._write_cache_metadata(metadata)

        self.assertEqual(metadata, store._read_cache_metadata())
        self.assertTrue(store._has_valid_cached_index())

    def test_retrieve_schema_context_caches_final_context(self) -> None:
        store = self.build_store_without_chroma()

        with patch("graphql_ai.rag.vector_store.embed_texts", return_value=[[0.1, 0.2]]):
            first_context = store.retrieve_schema_context("country")
            second_context = store.retrieve_schema_context("country")

        self.assertEqual(first_context, second_context)
        self.assertEqual("type Country: type Country { code: ID! }", first_context)
        self.assertEqual(1, store.collection.query_calls)
        self.assertEqual([4], store.collection.n_results)

    def test_format_schema_chunk_can_return_uncompressed_context(self) -> None:
        settings = AppSettings(
            schema_file=self.schema_file,
            chroma_path=self.settings.chroma_path,
            schema_context_cache_path=self.settings.schema_context_cache_path,
            prompt_compression_enabled=False,
        )
        store = self.build_store_without_chroma()
        store.settings = settings

        formatted = store._format_schema_chunk("type Country {\n  code: ID!\n}", "type", "Country")

        self.assertEqual("# type Country\ntype Country {\n  code: ID!\n}", formatted)


if __name__ == "__main__":
    unittest.main()
