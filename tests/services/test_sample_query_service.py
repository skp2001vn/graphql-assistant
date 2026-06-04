from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from graphql_ai.core.config import AppSettings
from graphql_ai.services.sample_query_service import (
    SampleQueryService,
    build_default_sample_request,
    parse_generated_sample,
    validate_operation_against_schema,
    validate_variable_usage,
)


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
  native: String!
  emoji: String!
  capital: String
  currency: String
  continent: Continent!
  languages: [Language!]!
}

type Continent {
  code: ID!
  name: String!
}

type Language {
  code: ID!
  name: String
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

    def retrieve_schema_context(self, user_request: str) -> str:
        self.requests.append(user_request)
        return self.context


class SampleQueryServiceTest(unittest.TestCase):
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

    def test_build_default_sample_request_for_country_is_specific(self) -> None:
        request = build_default_sample_request("country")

        self.assertIn("CountryQuery", request)
        self.assertIn("code", request)
        self.assertIn("US", request)

    def test_build_default_sample_request_for_other_target_requires_variables(self) -> None:
        request = build_default_sample_request("continent")

        self.assertIn("ContinentQuery", request)
        self.assertIn("define GraphQL variables", request)
        self.assertIn("do not hardcode", request)

    def test_build_default_sample_request_rejects_blank_target(self) -> None:
        with self.assertRaises(ValueError):
            build_default_sample_request("   ")

    def test_parse_generated_sample_reads_operation_and_variables_blocks(self) -> None:
        raw_response = """
```graphql
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
  }
}
```

```json
{"code": "US"}
```
"""

        sample = parse_generated_sample(raw_response)

        self.assertIn("CountryQuery", sample.operation)
        self.assertEqual({"code": "US"}, sample.variables)

    def test_parse_generated_sample_infers_variables_when_json_block_is_missing(self) -> None:
        raw_response = """
```graphql
query CountryQuery($code: ID!, $active: Boolean, $limit: Int) {
  country(code: $code) {
    code
  }
}
```
"""

        sample = parse_generated_sample(raw_response)

        self.assertEqual({"code": "US", "active": True, "limit": 1}, sample.variables)

    def test_validate_operation_rejects_field_missing_from_response_type(self) -> None:
        operation = """
query ContinentQuery($code: ID!) {
  continent(code: $code) {
    code
    name
    countries {
      code
    }
  }
}
"""

        errors = validate_operation_against_schema(operation, self.schema_file)

        self.assertEqual(["type Continent has no field countries"], errors)

    def test_validate_operation_accepts_nested_fields_that_exist(self) -> None:
        operation = """
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    continent {
      code
      name
    }
    languages {
      code
      name
    }
  }
}
"""

        errors = validate_operation_against_schema(operation, self.schema_file)

        self.assertEqual([], errors)

    def test_validate_variable_usage_rejects_unused_variables(self) -> None:
        errors = validate_variable_usage(
            'query ContinentQuery { continent(code: "AF") { code name } }',
            {"code": "AF"},
        )

        self.assertEqual(["variables JSON includes code, but operation does not use $code"], errors)

    def test_generate_uses_rag_context_llm_and_validates_output(self) -> None:
        llm_response = """
```graphql
query ContinentQuery($code: ID!) {
  continent(code: $code) {
    code
    name
  }
}
```

```json
{"code": "AF"}
```
"""
        llm_client = FakeLLMClient(llm_response)
        schema_context_provider = FakeSchemaContextProvider()
        service = SampleQueryService(
            settings=self.settings,
            llm_client=llm_client,
            schema_context_provider=schema_context_provider,
        )

        sample = service.generate("Generate continent query")

        self.assertEqual({"code": "AF"}, sample.variables)
        self.assertIn("Schema:", llm_client.prompts[0])
        self.assertEqual(["Generate continent query"], schema_context_provider.requests)

    def test_generate_raises_when_model_output_does_not_match_schema(self) -> None:
        llm_response = """
```graphql
query ContinentQuery($code: ID!) {
  continent(code: $code) {
    code
    countries {
      code
    }
  }
}
```

```json
{"code": "AF"}
```
"""
        service = SampleQueryService(
            settings=self.settings,
            llm_client=FakeLLMClient(llm_response),
            schema_context_provider=FakeSchemaContextProvider(),
        )

        with self.assertRaisesRegex(RuntimeError, "type Continent has no field countries"):
            service.generate("Generate continent query")

    def test_pre_warm_sends_configured_prompt_when_enabled(self) -> None:
        llm_client = FakeLLMClient("ok")
        service = SampleQueryService(
            settings=self.settings,
            llm_client=llm_client,
            schema_context_provider=FakeSchemaContextProvider(),
        )

        service.pre_warm()

        self.assertEqual(["warm"], llm_client.prompts)

    def test_pre_warm_is_skipped_when_disabled(self) -> None:
        settings = AppSettings(schema_file=self.schema_file, ollama_pre_warm_enabled=False)
        llm_client = FakeLLMClient("ok")
        service = SampleQueryService(
            settings=settings,
            llm_client=llm_client,
            schema_context_provider=FakeSchemaContextProvider(),
        )

        service.pre_warm()

        self.assertEqual([], llm_client.prompts)


if __name__ == "__main__":
    unittest.main()
