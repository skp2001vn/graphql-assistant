from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_ai.agents.troubleshooting_agent import (
    GraphQLValidationTool,
    TroubleshootingAgent,
    parse_troubleshooting_response,
)
from graphql_ai.core.config import AppSettings
from graphql_ai.services.sample_query_service import InvalidRootFieldNameError


SCHEMA = """
type Query {
  countries: [Country!]!
  country(code: ID!): Country
  continents: [Continent!]!
  continent(code: ID!): Continent
}

type Country {
  code: ID!
  name: String!
}

type Continent {
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


class FakeSchemaContextProvider:
    def __init__(self, context: str = SCHEMA) -> None:
        self.context = context
        self.requests: list[str] = []

    def retrieve_schema_context(self, retrieval_request: str) -> str:
        self.requests.append(retrieval_request)
        return self.context


class TroubleshootingAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.schema_file = Path(self.temp_dir.name) / "schema.graphql"
        self.schema_file.write_text(SCHEMA, encoding="utf-8")
        self.settings = AppSettings(
            schema_file=self.schema_file,
            inference_cache_enabled=False,
            ollama_pre_warm_enabled=False,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validation_tool_reports_typo_suggestions(self) -> None:
        tool = GraphQLValidationTool(self.schema_file)

        observation = tool.validate(
            """
query CountryQuery($code: ID!) {
  county(code: $code) {
    code
  }
}
"""
        )

        self.assertEqual(1, len(observation.issues))
        self.assertIn("Cannot query field 'county' on type 'Query'", observation.issues[0])
        self.assertIn("Did you mean 'country'", observation.issues[0])

    def test_validation_tool_reports_syntax_location(self) -> None:
        tool = GraphQLValidationTool(self.schema_file)

        observation = tool.validate("query CountryQuery($code: ID!) { country(code: $code) { code ")

        self.assertEqual(1, len(observation.issues))
        self.assertIn("Syntax Error", observation.issues[0])
        self.assertIn("Location: line", observation.issues[0])

    def test_parse_troubleshooting_response_reads_suggestion_and_corrected_operation(self) -> None:
        suggestion, corrected_operation = parse_troubleshooting_response(
            """
```text
Use the schema field `country` instead of `county`.
```

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

        self.assertIn("country", suggestion)
        self.assertIn("country(code: $code)", corrected_operation)

    def test_agent_runs_plan_tools_and_validates_corrected_operation(self) -> None:
        llm_response = """
```text
Use the schema field `country` instead of `county`.
```

```graphql
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
  }
}
```
"""
        llm_client = FakeLLMClient(llm_response)
        schema_context_provider = FakeSchemaContextProvider()
        agent = TroubleshootingAgent(
            settings=self.settings,
            llm_client=llm_client,
            schema_context_provider=schema_context_provider,
        )

        result = agent.troubleshoot(
            "county",
            """
query CountyQuery($code: ID!) {
  county(code: $code) {
    code
  }
}
""",
        )

        self.assertEqual("county", result.root_field)
        self.assertEqual("invalid", result.status)
        self.assertIn("Cannot query field 'county'", result.issues[0])
        self.assertIn("Use the schema field", result.suggestion)
        self.assertIn("country(code: $code)", result.corrected_operation)
        self.assertEqual(
            ["Troubleshoot GraphQL Query or Mutation root field county"],
            schema_context_provider.requests,
        )
        self.assertIn("Plan:", llm_client.prompts[0])
        self.assertIn("Validation issues:", llm_client.prompts[0])

    def test_agent_rejects_invalid_root_field_before_tools(self) -> None:
        llm_client = FakeLLMClient("unused")
        schema_context_provider = FakeSchemaContextProvider()
        agent = TroubleshootingAgent(
            settings=self.settings,
            llm_client=llm_client,
            schema_context_provider=schema_context_provider,
        )

        with self.assertRaisesRegex(InvalidRootFieldNameError, "GraphQL field name"):
            agent.troubleshoot("123bad", "query CountryQuery { country(code: \"US\") { code } }")

        self.assertEqual([], llm_client.prompts)
        self.assertEqual([], schema_context_provider.requests)

    def test_agent_drops_invalid_corrected_operation(self) -> None:
        llm_response = """
```text
Use the correct schema field.
```

```graphql
query CountryQuery {
  country {
    missing
  }
}
```
"""
        agent = TroubleshootingAgent(
            settings=self.settings,
            llm_client=FakeLLMClient(llm_response),
            schema_context_provider=FakeSchemaContextProvider(),
        )

        result = agent.troubleshoot(
            "county",
            "query CountyQuery { county { code } }",
        )

        self.assertEqual("", result.corrected_operation)
        self.assertTrue(any("Corrected operation was still invalid" in issue for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
