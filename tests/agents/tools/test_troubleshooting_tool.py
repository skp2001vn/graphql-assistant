from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_assistant.core.config import AppSettings
from graphql_assistant.agents.tools import RootFieldNotInSchemaError
from graphql_assistant.agents.tools.troubleshooting_tool import TroubleshootingTool


SCHEMA = """
type Query {
  country(code: ID!): Country
}

type Country {
  code: ID!
  name: String!
}
"""


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeLLMPreWarmer:
    def __init__(self) -> None:
        self.calls = 0

    def pre_warm(self) -> None:
        self.calls += 1


class FakeSchemaContextProvider:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def retrieve_schema_context(self, retrieval_request: str) -> str:
        self.requests.append(retrieval_request)
        return SCHEMA


class TroubleshootingToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.schema_file = Path(self.temp_dir.name) / "schema.graphql"
        self.schema_file.write_text(SCHEMA, encoding="utf-8")
        self.settings = AppSettings(
            schema_file=self.schema_file,
            inference_cache_enabled=False,
            ollama_pre_warm_enabled=True,
            ollama_pre_warm_prompt="warm",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_troubleshoot_prewarms_before_inference_for_invalid_call(self) -> None:
        llm_client = FakeLLMClient(
            """
DETAIL:
```text
Use the schema root field country.
```

SUGGESTION:
```graphql
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
  }
}
```
"""
        )
        pre_warmer = FakeLLMPreWarmer()
        schema_context_provider = FakeSchemaContextProvider()
        tool = TroubleshootingTool(
            settings=self.settings,
            llm_client=llm_client,
            llm_pre_warmer=pre_warmer,
            schema_context_provider=schema_context_provider,
        )

        result = tool.troubleshoot(
            "country",
            'query CountyQuery($code: ID!) { county(code: $code) { code } }',
        )

        self.assertEqual(1, pre_warmer.calls)
        self.assertEqual(1, len(llm_client.prompts))
        self.assertIn("country", result.suggestion)
        self.assertEqual(["Troubleshoot GraphQL Query or Mutation root field country"], schema_context_provider.requests)

    def test_troubleshoot_skips_prewarm_when_call_is_already_valid(self) -> None:
        llm_client = FakeLLMClient("unused")
        pre_warmer = FakeLLMPreWarmer()
        schema_context_provider = FakeSchemaContextProvider()
        tool = TroubleshootingTool(
            settings=self.settings,
            llm_client=llm_client,
            llm_pre_warmer=pre_warmer,
            schema_context_provider=schema_context_provider,
        )

        result = tool.troubleshoot(
            "country",
            'query CountryQuery($code: ID!) { country(code: $code) { code name } }',
        )

        self.assertEqual("valid", result.status)
        self.assertEqual(0, pre_warmer.calls)
        self.assertEqual([], llm_client.prompts)
        self.assertEqual([], schema_context_provider.requests)

    def test_troubleshoot_rejects_root_field_missing_from_schema(self) -> None:
        llm_client = FakeLLMClient("unused")
        pre_warmer = FakeLLMPreWarmer()
        schema_context_provider = FakeSchemaContextProvider()
        tool = TroubleshootingTool(
            settings=self.settings,
            llm_client=llm_client,
            llm_pre_warmer=pre_warmer,
            schema_context_provider=schema_context_provider,
        )

        with self.assertRaisesRegex(RootFieldNotInSchemaError, "No GraphQL Query or Mutation field named `city`"):
            tool.troubleshoot(
                "city",
                'query CountryQuery($code: ID!) { country(code: $code) { code name } }',
            )

        self.assertEqual(0, pre_warmer.calls)
        self.assertEqual([], llm_client.prompts)
        self.assertEqual([], schema_context_provider.requests)


if __name__ == "__main__":
    unittest.main()
