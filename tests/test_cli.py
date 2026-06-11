from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from graphql_ai import cli
from graphql_ai.domain import GeneratedGraphQLSample


class FakeLLMPreWarmer:
    def __init__(self) -> None:
        self.pre_warm_called = False

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeSampleQueryTool:
    def __init__(self, rebuild_index: bool = False) -> None:
        self.rebuild_index = rebuild_index
        self.llm_pre_warmer = FakeLLMPreWarmer()

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        return GeneratedGraphQLSample(
            operation="query Test { countries { code } }",
            variables={},
            raw_response=f"raw response for {root_field}; rebuild={self.rebuild_index}",
        )


class CliTest(unittest.TestCase):
    def test_parse_args_uses_default_root_field(self) -> None:
        with patch("sys.argv", ["graphql-ai"]):
            args = cli.parse_args()

        self.assertEqual("country", args.root_field)
        self.assertFalse(args.rebuild)

    def test_main_prints_generated_raw_response(self) -> None:
        output = io.StringIO()

        with patch("sys.argv", ["graphql-ai", "--rebuild", "countries"]):
            with patch("graphql_ai.cli.SampleQueryTool", FakeSampleQueryTool):
                with redirect_stdout(output):
                    cli.main()

        self.assertIn("Generated result:", output.getvalue())
        self.assertIn("countries", output.getvalue())
        self.assertIn("rebuild=True", output.getvalue())


if __name__ == "__main__":
    unittest.main()
