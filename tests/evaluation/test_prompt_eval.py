from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_assistant.evaluation import prompt_eval
from graphql_assistant.evaluation.prompt_eval import (
    SamplePromptEvalCase,
    TroubleshootingPromptEvalCase,
    run_sample_prompt_eval_cases,
    run_troubleshooting_prompt_eval_cases,
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


class FakeSampleTool:
    def __init__(self, sample: GeneratedGraphQLSample, schema_file: Path | None = None) -> None:
        self.sample = sample
        self.settings = type("Settings", (), {"schema_file": schema_file})()
        self.root_fields: list[str] = []

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        self.root_fields.append(root_field)
        return self.sample


class FakeTroubleshootingTool:
    def __init__(self, result: TroubleshootingResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        self.calls.append((root_field, graphql_call))
        return self.result


class FakeLLMPreWarmer:
    def __init__(self) -> None:
        self.pre_warm_called = False

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeMainSampleTool:
    def __init__(self, rebuild_index: bool = False) -> None:
        self.settings = type("Settings", (), {"schema_file": Path("schema.graphql")})()
        self.llm_client = object()
        self.llm_pre_warmer = FakeLLMPreWarmer()
        self.schema_context_provider = object()
        self.rebuild_index = rebuild_index


class FakeMainTroubleshootingTool:
    def __init__(self, **_: object) -> None:
        pass


class PromptEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.schema_file = Path(self.temp_dir.name) / "schema.graphql"
        self.schema_file.write_text(SCHEMA, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sample_eval_passes_valid_generation(self) -> None:
        sample = GeneratedGraphQLSample(
            operation="""
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
  }
}
""",
            variables={"code": "US"},
            raw_response="raw",
        )
        tool = FakeSampleTool(sample, self.schema_file)
        cases = [
            SamplePromptEvalCase(
                name="country",
                root_field="country",
                expected_text=("country(code:", "code", "name"),
            )
        ]

        results = run_sample_prompt_eval_cases(tool, cases)

        self.assertTrue(results[0].passed)
        self.assertEqual(["country"], tool.root_fields)
        self.assertIn("PASS operation validates against schema", results[0].checks)

    def test_sample_eval_fails_invalid_generation(self) -> None:
        sample = GeneratedGraphQLSample(
            operation="query CountryQuery($code: ID!) { country(code: $code) { code1 } }",
            variables={"code": "US"},
            raw_response="raw",
        )
        tool = FakeSampleTool(sample, self.schema_file)
        cases = [SamplePromptEvalCase(name="country", root_field="country", expected_text=("code",))]

        results = run_sample_prompt_eval_cases(tool, cases)

        self.assertFalse(results[0].passed)
        self.assertIn("FAIL operation validates against schema", results[0].checks[0])

    def test_troubleshooting_eval_passes_fixed_suggestion(self) -> None:
        result = TroubleshootingResult(
            root_field="country",
            status="invalid",
            issues=["Cannot query field 'code1' on type 'Country'."],
            detail=["Use `code` instead of `code1`."],
            suggestion="""
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
  }
}
""",
            raw_response="raw",
        )
        tool = FakeTroubleshootingTool(result)
        cases = [
            TroubleshootingPromptEvalCase(
                name="field typo",
                root_field="country",
                graphql_call="query CountryQuery($code: ID!) { country(code: $code) { code1 } }",
                expected_suggestion_text=("country(code:", "code", "name"),
            )
        ]

        results = run_troubleshooting_prompt_eval_cases(tool, self.schema_file, cases)

        self.assertTrue(results[0].passed)
        self.assertEqual([("country", cases[0].graphql_call)], tool.calls)
        self.assertIn("PASS suggestion validates against schema", results[0].checks)

    def test_main_prints_summary(self) -> None:
        output = io.StringIO()
        result = prompt_eval.PromptEvalResult("sample", "case", True, ("PASS check",))

        with patch("sys.argv", ["prompt-eval", "--workflow", "sample"]):
            with patch("graphql_assistant.evaluation.prompt_eval.SampleTool", FakeMainSampleTool):
                with patch("graphql_assistant.evaluation.prompt_eval.TroubleshootingTool", FakeMainTroubleshootingTool):
                    with patch("graphql_assistant.evaluation.prompt_eval.run_sample_prompt_eval_cases", return_value=[result]):
                        with redirect_stdout(output):
                            prompt_eval.main()

        self.assertIn("[PASS] sample: case", output.getvalue())
        self.assertIn("Summary: 1/1 passed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
