from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_ai.agents.troubleshooting_agent import (
    GraphQLValidationTool,
    TroubleshootingAgent,
    clean_model_detail,
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
    def __init__(self, response: str | list[str]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.responses) == 1:
            return self.responses[0]

        return self.responses.pop(0)


class FakeSchemaContextProvider:
    def __init__(self, context: str = SCHEMA) -> None:
        self.context = context
        self.requests: list[str] = []

    def retrieve_schema_context(self, retrieval_request: str) -> str:
        self.requests.append(retrieval_request)
        return self.context


class CountingSchemaFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.read_count = 0

    def read_text(self, encoding: str) -> str:
        self.read_count += 1
        return self.path.read_text(encoding=encoding)


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

        issues = tool.validate(
            """
query CountryQuery($code: ID!) {
  county(code: $code) {
    code
  }
}
"""
        )

        self.assertEqual(1, len(issues))
        self.assertIn("Cannot query field 'county' on type 'Query'", issues[0])
        self.assertIn("Did you mean 'country'", issues[0])

    def test_validation_tool_reports_syntax_location(self) -> None:
        tool = GraphQLValidationTool(self.schema_file)

        issues = tool.validate("query CountryQuery($code: ID!) { country(code: $code) { code ")

        self.assertEqual(1, len(issues))
        self.assertIn("Syntax Error", issues[0])
        self.assertIn("Location: line", issues[0])

    def test_validation_tool_caches_parsed_schema(self) -> None:
        schema_file = CountingSchemaFile(self.schema_file)
        tool = GraphQLValidationTool(schema_file)

        tool.validate("query CountryQuery($code: ID!) { country(code: $code) { code } }")
        tool.validate("query CountriesQuery { countries { code } }")

        self.assertEqual(1, schema_file.read_count)

    def test_parse_troubleshooting_response_reads_detail_and_suggested_operation(self) -> None:
        detail, suggestion = parse_troubleshooting_response(
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

        self.assertIn("country", detail[0])
        self.assertIn("country(code: $code)", suggestion)

    def test_parse_troubleshooting_response_prefers_labeled_blocks(self) -> None:
        detail, suggestion = parse_troubleshooting_response(
            """
DETAIL:
```text
Replace `name1` with `name` in the selected fields.
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

        self.assertEqual(["Replace `name1` with `name` in the selected fields."], detail)
        self.assertIn("name", suggestion)

    def test_clean_model_detail_removes_raw_issues(self) -> None:
        detail = [
            "1. **Cannot query field 'coe' on type 'Country'. Did you mean 'code'?**",
            "- Change `coe` to `code` in the submitted operation.",
        ]
        issues = ["Cannot query field 'coe' on type 'Country'. Did you mean 'code'? Location: line 3, column 5."]

        self.assertEqual(
            ["Change `coe` to `code` in the submitted operation."],
            clean_model_detail(detail, issues),
        )

    def test_agent_runs_plan_tools_and_validates_suggested_operation(self) -> None:
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
        self.assertIn("Use the schema field", result.detail[0])
        self.assertIn("country(code: $code)", result.suggestion)
        self.assertEqual(
            ["Troubleshoot GraphQL Query or Mutation root field county"],
            schema_context_provider.requests,
        )
        self.assertIn("Plan:", llm_client.prompts[0])
        self.assertIn("Validation issues:", llm_client.prompts[0])
        self.assertIn("Detail block rules:", llm_client.prompts[0])
        self.assertIn("Do not include headings, bullets, numbering, JSON, or GraphQL code", llm_client.prompts[0])
        self.assertIn("DETAIL:", llm_client.prompts[0])
        self.assertIn("SUGGESTION:", llm_client.prompts[0])
        self.assertIn("Do not copy placeholder text", llm_client.prompts[0])
        self.assertIn("For syntax errors, fix only GraphQL structure", llm_client.prompts[0])
        self.assertIn("Preserve submitted fields that are not named in a validation issue", llm_client.prompts[0])
        self.assertIn("Do not replace a nested object field with a root Query field", llm_client.prompts[0])

    def test_agent_caches_retrieved_schema_context_per_root_field(self) -> None:
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
        schema_context_provider = FakeSchemaContextProvider()
        agent = TroubleshootingAgent(
            settings=self.settings,
            llm_client=FakeLLMClient(llm_response),
            schema_context_provider=schema_context_provider,
        )

        agent.troubleshoot("county", "query CountyQuery($code: ID!) { county(code: $code) { code } }")
        agent.troubleshoot("county", "query CountyQuery($code: ID!) { county(code: $code) { name } }")

        self.assertEqual(
            ["Troubleshoot GraphQL Query or Mutation root field county"],
            schema_context_provider.requests,
        )

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

    def test_agent_returns_empty_fields_for_valid_graphql_call(self) -> None:
        llm_client = FakeLLMClient("unused")
        schema_context_provider = FakeSchemaContextProvider()
        agent = TroubleshootingAgent(
            settings=self.settings,
            llm_client=llm_client,
            schema_context_provider=schema_context_provider,
        )

        result = agent.troubleshoot(
            "country",
            """
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
  }
}
""",
        )

        self.assertEqual("valid", result.status)
        self.assertEqual([], result.issues)
        self.assertEqual([], result.detail)
        self.assertEqual("", result.suggestion)
        self.assertEqual("", result.raw_response)
        self.assertEqual([], llm_client.prompts)
        self.assertEqual([], schema_context_provider.requests)

    def test_agent_drops_invalid_suggested_operation(self) -> None:
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

        self.assertEqual("", result.suggestion)
        self.assertTrue(any("Corrected operation was still invalid" in issue for issue in result.issues))

    def test_agent_does_not_retry_when_first_response_returns_only_operation(self) -> None:
        llm_response = """
```graphql
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
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
            "query CountyQuery($code: ID!) { county(code: $code) { code } }",
        )

        self.assertEqual([], result.detail)
        self.assertIn("country(code: $code)", result.suggestion)
        self.assertEqual(1, len(agent.llm_client.prompts))

    def test_agent_does_not_retry_when_model_repeats_validation_issue(self) -> None:
        raw_issue = "Cannot query field 'county' on type 'Query'. Did you mean 'country'?"
        llm_response = f"""
```text
- {raw_issue}
```

```graphql
query CountryQuery($code: ID!) {{
  country(code: $code) {{
    code
    name
  }}
}}
```
"""
        agent = TroubleshootingAgent(
            settings=self.settings,
            llm_client=FakeLLMClient(llm_response),
            schema_context_provider=FakeSchemaContextProvider(),
        )

        result = agent.troubleshoot(
            "county",
            "query CountyQuery($code: ID!) { county(code: $code) { code } }",
        )

        self.assertEqual([], result.detail)
        self.assertIn("country(code: $code)", result.suggestion)
        self.assertEqual(1, len(agent.llm_client.prompts))


if __name__ == "__main__":
    unittest.main()
