from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from graphql_ai import cli
from graphql_ai.domain import GeneratedGraphQLSample


class FakeSampleService:
    def __init__(self, rebuild_index: bool = False) -> None:
        self.rebuild_index = rebuild_index

    def generate(self, request: str) -> GeneratedGraphQLSample:
        return GeneratedGraphQLSample(
            operation="query Test { countries { code } }",
            variables={},
            raw_response=f"raw response for {request}; rebuild={self.rebuild_index}",
        )


class CliTest(unittest.TestCase):
    def test_parse_args_uses_default_request(self) -> None:
        with patch("sys.argv", ["graphql-ai"]):
            args = cli.parse_args()

        self.assertEqual("Generate a sample query for a country by code", args.request)
        self.assertFalse(args.rebuild)

    def test_main_prints_generated_raw_response(self) -> None:
        output = io.StringIO()

        with patch("sys.argv", ["graphql-ai", "--rebuild", "custom request"]):
            with patch("graphql_ai.cli.SampleQueryService", FakeSampleService):
                with redirect_stdout(output):
                    cli.main()

        self.assertIn("Generated result:", output.getvalue())
        self.assertIn("custom request", output.getvalue())
        self.assertIn("rebuild=True", output.getvalue())


if __name__ == "__main__":
    unittest.main()
