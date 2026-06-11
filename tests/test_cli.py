from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from graphql_assistant import cli
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult


class FakeLLMPreWarmer:
    def __init__(self) -> None:
        self.pre_warm_called = False

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeAssistantAgent:
    def __init__(
        self,
        sample_tool: object,
        troubleshooting_tool: object,
        planner: object,
    ) -> None:
        self.sample_tool = sample_tool
        self.troubleshooting_tool = troubleshooting_tool
        self.planner = planner
        self.goals = []

    def run(self, goal: object) -> object:
        self.goals.append(goal)
        if getattr(goal, "graphql_call", None):
            return type(
                "Result",
                (),
                {
                    "intent": "troubleshoot",
                    "output": TroubleshootingResult(
                        root_field=goal.root_field,
                        status="invalid",
                        issues=["bad query"],
                        detail=["Fix the field name."],
                        suggestion="query CountryQuery { country { code } }",
                        raw_response="raw troubleshoot",
                    ),
                },
            )()
        return type(
            "Result",
            (),
            {
                "intent": "generate_sample",
                "output": GeneratedGraphQLSample(
                    operation="query Test { countries { code } }",
                    variables={},
                    raw_response=f"raw response for {goal.root_field}",
                ),
            },
        )()


class CliTest(unittest.TestCase):
    def test_parse_args_uses_default_root_field(self) -> None:
        with patch("sys.argv", ["graphql-ai"]):
            args = cli.parse_args()

        self.assertEqual("Generate a sample query", args.goal)
        self.assertEqual("country", args.root_field)
        self.assertIsNone(args.graphql_call)
        self.assertFalse(args.rebuild)

    def test_main_prints_generated_raw_response(self) -> None:
        output = io.StringIO()

        with (
            patch("sys.argv", ["graphql-ai", "--rebuild", "Generate a sample query", "countries"]),
            patch("graphql_assistant.cli.get_settings", return_value=object()),
            patch("graphql_assistant.cli.SchemaVectorStore", return_value=object()),
            patch("graphql_assistant.cli.build_llm_client", return_value=object()),
            patch("graphql_assistant.cli.LLMPreWarmer", return_value=FakeLLMPreWarmer()),
            patch("graphql_assistant.cli.AgnoAssistantPlanner", return_value=object()),
            patch("graphql_assistant.cli.SampleTool", return_value=object()),
            patch("graphql_assistant.cli.TroubleshootingTool", return_value=object()),
            patch("graphql_assistant.cli.GraphQLAssistantAgent", FakeAssistantAgent),
        ):
            with redirect_stdout(output):
                cli.main()

        self.assertIn("Assistant intent: generate_sample", output.getvalue())
        self.assertIn("Generated result:", output.getvalue())
        self.assertIn("countries", output.getvalue())

    def test_main_prints_troubleshooting_result(self) -> None:
        output = io.StringIO()

        with (
            patch(
                "sys.argv",
                [
                    "graphql-ai",
                    "Troubleshoot this operation",
                    "country",
                    "--graphql-call",
                    "query CountryQuery { country { code1 } }",
                ],
            ),
            patch("graphql_assistant.cli.get_settings", return_value=object()),
            patch("graphql_assistant.cli.SchemaVectorStore", return_value=object()),
            patch("graphql_assistant.cli.build_llm_client", return_value=object()),
            patch("graphql_assistant.cli.LLMPreWarmer", return_value=FakeLLMPreWarmer()),
            patch("graphql_assistant.cli.AgnoAssistantPlanner", return_value=object()),
            patch("graphql_assistant.cli.SampleTool", return_value=object()),
            patch("graphql_assistant.cli.TroubleshootingTool", return_value=object()),
            patch("graphql_assistant.cli.GraphQLAssistantAgent", FakeAssistantAgent),
        ):
            with redirect_stdout(output):
                cli.main()

        self.assertIn("Assistant intent: troubleshoot", output.getvalue())
        self.assertIn("Troubleshooting result:", output.getvalue())
        self.assertIn("Status: invalid", output.getvalue())
        self.assertIn("Fix the field name.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
