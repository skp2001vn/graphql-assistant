from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from graphql_assistant.agents import AgentPlanningError, GraphQLAssistantGoal, GraphQLAssistantResult
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_assistant.evaluation import prompt_eval
from graphql_assistant.evaluation.prompt_eval import (
    AssistantPromptEvalCase,
    SamplePromptEvalCase,
    TroubleshootingPromptEvalCase,
    run_assistant_prompt_eval_cases,
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


class FakeAssistant:
    def __init__(self, response: GraphQLAssistantResult | Exception) -> None:
        self.response = response
        self.goals: list[GraphQLAssistantGoal] = []

    def run(self, goal: GraphQLAssistantGoal) -> GraphQLAssistantResult:
        self.goals.append(goal)
        if isinstance(self.response, Exception):
            raise self.response

        return self.response


class FakeLLMPreWarmer:
    def __init__(self) -> None:
        self.pre_warm_called = False

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeMainSampleTool:
    def __init__(self, rebuild_index: bool = False) -> None:
        self.settings = type("Settings", (), {"schema_file": Path("schema.graphql")})()
        self.llm_client = object()
        self.schema_context_provider = object()
        self.rebuild_index = rebuild_index


class FakeMainTroubleshootingTool:
    def __init__(self, **_: object) -> None:
        pass


class FakeMainAssistant:
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

    def test_sample_eval_fails_when_generation_targets_wrong_root_field(self) -> None:
        sample = GeneratedGraphQLSample(
            operation="""
query CountriesQuery {
  countries {
    code
    name
  }
}
""",
            variables={},
            raw_response="raw",
        )
        tool = FakeSampleTool(sample, self.schema_file)
        cases = [
            SamplePromptEvalCase(
                name="country",
                root_field="country",
                expected_text=("code", "name"),
            )
        ]

        results = run_sample_prompt_eval_cases(tool, cases)

        self.assertFalse(results[0].passed)
        self.assertTrue(
            any(check.startswith("FAIL operation targets requested root field `country`") for check in results[0].checks)
        )

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

    def test_troubleshooting_eval_fails_when_status_conflicts_with_issues(self) -> None:
        result = TroubleshootingResult(
            root_field="country",
            status="valid",
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

        self.assertFalse(results[0].passed)
        self.assertTrue(
            any(check.startswith("FAIL result status matches troubleshooting outcome") for check in results[0].checks)
        )

    def test_assistant_eval_passes_sample_request(self) -> None:
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
        case = AssistantPromptEvalCase(
            name="assistant sample",
            goal="Generate a sample query",
            root_field="country",
            expected_intent="generate_sample",
            expected_text=("country(code:", "code", "name"),
        )
        assistant = FakeAssistant(
            GraphQLAssistantResult(
                intent="generate_sample",
                goal=GraphQLAssistantGoal(goal=case.goal, root_field=case.root_field),
                output=sample,
                raw_plan_response='{"intent":"generate_sample"}',
            )
        )

        results = run_assistant_prompt_eval_cases(assistant, self.schema_file, [case])

        self.assertTrue(results[0].passed)
        self.assertEqual([GraphQLAssistantGoal(goal=case.goal, root_field=case.root_field, graphql_call=None)], assistant.goals)
        self.assertIn("PASS assistant selects `generate_sample` intent", results[0].checks)

    def test_assistant_eval_passes_unsupported_goal(self) -> None:
        case = AssistantPromptEvalCase(
            name="unsupported",
            goal="sdfdsfdf",
            root_field="country",
            expected_intent="unsupported",
            expected_error_text="Assistant goal must ask to generate a sample GraphQL operation or troubleshoot a GraphQL operation.",
        )
        assistant = FakeAssistant(AgentPlanningError(case.expected_error_text))

        results = run_assistant_prompt_eval_cases(assistant, self.schema_file, [case])

        self.assertTrue(results[0].passed)
        self.assertIn("PASS assistant rejects unsupported goal", results[0].checks)

    def test_assistant_eval_raises_for_invalid_return_type(self) -> None:
        case = AssistantPromptEvalCase(
            name="bad assistant contract",
            goal="Generate a sample query",
            root_field="country",
            expected_intent="generate_sample",
        )

        class InvalidAssistant:
            def run(self, goal: GraphQLAssistantGoal) -> str:
                return "not-a-result"

        with self.assertRaises(TypeError):
            run_assistant_prompt_eval_cases(InvalidAssistant(), self.schema_file, [case])

    def test_main_prints_summary(self) -> None:
        output = io.StringIO()
        result = prompt_eval.PromptEvalResult("assistant", "case", True, ("PASS check",))

        with patch("sys.argv", ["prompt-eval", "--intent", "generate_sample"]):
            with patch("graphql_assistant.evaluation.prompt_eval.SampleTool", FakeMainSampleTool):
                with patch("graphql_assistant.evaluation.prompt_eval.TroubleshootingTool", FakeMainTroubleshootingTool):
                    with patch("graphql_assistant.evaluation.prompt_eval.LLMPreWarmer", return_value=FakeLLMPreWarmer()):
                        with patch("graphql_assistant.evaluation.prompt_eval.AgnoAssistantPlanner", return_value=object()):
                            with patch("graphql_assistant.evaluation.prompt_eval.GraphQLAssistantAgent", FakeMainAssistant):
                                with patch("graphql_assistant.evaluation.prompt_eval.run_assistant_prompt_eval_cases", return_value=[result]):
                                    with redirect_stdout(output):
                                        prompt_eval.main()

        self.assertIn("[PASS] assistant: case", output.getvalue())
        self.assertIn("Summary: 1/1 passed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
