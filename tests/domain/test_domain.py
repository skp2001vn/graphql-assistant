from __future__ import annotations

import unittest

from graphql_assistant.domain import GeneratedGraphQLSample, SchemaChunk


class DomainTest(unittest.TestCase):
    def test_schema_chunk_is_immutable_domain_value(self) -> None:
        chunk = SchemaChunk(id="1", source="schema.graphql", kind="type", name="Country", text="type Country")

        self.assertEqual("Country", chunk.name)
        with self.assertRaises(Exception):
            chunk.name = "Other"  # type: ignore[misc]

    def test_generated_sample_holds_operation_variables_and_raw_response(self) -> None:
        sample = GeneratedGraphQLSample(operation="query X", variables={"code": "US"}, raw_response="raw")

        self.assertEqual("query X", sample.operation)
        self.assertEqual({"code": "US"}, sample.variables)
        self.assertEqual("raw", sample.raw_response)


if __name__ == "__main__":
    unittest.main()
