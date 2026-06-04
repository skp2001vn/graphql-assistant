from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_ai.rag.schema_chunks import (
    chunk_graphql_schema,
    load_schema_chunks,
    read_schema_file,
)


SCHEMA = """
type Query {
  countries: [Country!]!
  country(code: ID!): Country
}

type Country {
  code: ID!
  name: String!
}
"""


class SchemaChunksTest(unittest.TestCase):
    def test_read_schema_file_returns_trimmed_schema_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "schema.graphql"
            schema_file.write_text(f"\n{SCHEMA}\n", encoding="utf-8")

            source, schema_text = read_schema_file(schema_file)

        self.assertEqual(str(schema_file), source)
        self.assertEqual(SCHEMA.strip(), schema_text)

    def test_read_schema_file_rejects_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_schema_file(Path("missing-schema.graphql"))

    def test_read_schema_file_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "schema.graphql"
            schema_file.write_text("   ", encoding="utf-8")

            with self.assertRaises(ValueError):
                read_schema_file(schema_file)

    def test_chunk_graphql_schema_splits_top_level_definitions(self) -> None:
        chunks = chunk_graphql_schema(SCHEMA, "schema.graphql")

        chunk_keys = {(chunk.kind, chunk.name) for chunk in chunks}
        self.assertEqual(
            {
                ("type", "Query"),
                ("type", "Country"),
            },
            chunk_keys,
        )

    def test_chunk_graphql_schema_falls_back_to_file_chunk_for_unknown_format(self) -> None:
        chunks = chunk_graphql_schema("not SDL", "schema.graphql")

        self.assertEqual(1, len(chunks))
        self.assertEqual("file", chunks[0].kind)
        self.assertEqual("not SDL", chunks[0].text)

    def test_load_schema_chunks_rejects_empty_chunk_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "schema.graphql"
            schema_file.write_text(SCHEMA, encoding="utf-8")

            chunks = load_schema_chunks(schema_file)

        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
