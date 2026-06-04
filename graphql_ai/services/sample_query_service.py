from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

from graphql_ai.core.config import AppSettings, get_settings
from graphql_ai.core.protocols import SchemaContextProvider
from graphql_ai.domain import GeneratedGraphQLSample
from graphql_ai.llm.base import LLMClient
from graphql_ai.llm.cache import CachedLLMClient, PromptResponseCache
from graphql_ai.llm.ollama_client import OllamaClient
from graphql_ai.rag.vector_store import SchemaVectorStore


GRAPHQL_SYSTEM_PROMPT = (
    "You are a GraphQL expert. Use only schema fields. Include all fields for the requested "
    "response type, expand nested objects/lists only when those fields exist on that type, "
    "do not infer reverse relationship fields from root queries or other types, use variables, "
    "put sample values only in the variables JSON and never hardcode them in arguments, "
    "and return exactly two fenced code blocks: GraphQL operation, then variables JSON."
)

GRAPHQL_USER_PROMPT_TEMPLATE = "Schema:\n{schema_context}\n\nRequest:\n{user_request}"

VARIABLE_DEFINITION = re.compile(r"\$([_A-Za-z][_0-9A-Za-z]*)\s*:\s*([!\[\]_0-9A-Za-z]+)")


def build_default_sample_request(target: str) -> str:
    """Build the default natural-language request for a sample GraphQL target."""
    normalized_target = target.strip().replace("-", " ")
    if not normalized_target:
        raise ValueError("Target must not be empty.")

    target_name = normalized_target.title().replace(" ", "")
    if normalized_target.lower() == "country":
        return (
            "Generate a sample GraphQL query named CountryQuery for a country by code. "
            "Use a variable named code with type ID. "
            "Include all available Country fields, including continent and languages. "
            "Return Variables JSON with code set to US."
        )

    return (
        f"Generate a sample GraphQL query named {target_name}Query for {normalized_target}. "
        "For required arguments, define GraphQL variables in the operation signature and pass "
        "those variables into the field call. "
        "Include only fields that exist directly on the selected response type, then expand nested "
        "object or list fields only when the schema defines them. "
        "Return Variables JSON with realistic sample values, and do not hardcode those values in "
        "the GraphQL operation."
    )


class SampleQueryService:
    """Business service for generating sample GraphQL queries.

    This service coordinates the application workflow for the sample-query use case:
    it receives a natural-language request, retrieves schema context through the
    configured schema-context provider, sends a prompt to the configured LLM
    client, and parses the model output into a GraphQL operation plus variables.

    The current default schema-context provider is RAG-backed: `SchemaVectorStore`
    chunks the local GraphQL SDL, embeds the chunks, stores them in Chroma, and
    retrieves relevant schema context for each request. The service depends on
    the `SchemaContextProvider` protocol, so that RAG can later be replaced or
    composed with other approaches such as agent workflows or inference
    optimization without changing the API layer.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        schema_context_provider: SchemaContextProvider | None = None,
        rebuild_index: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        """Create a sample-query service with injectable LLM and schema context dependencies."""
        self.settings = settings or get_settings()
        self.schema_context_provider = schema_context_provider or SchemaVectorStore(
            settings=self.settings,
            rebuild=rebuild_index,
            allow_downloads=allow_downloads,
        )
        self.llm_client = llm_client or self._build_default_llm_client()
        self._generation_lock = Lock()

    def generate(self, user_request: str) -> GeneratedGraphQLSample:
        """Generate a sample GraphQL operation and variables for a user request.

        The prompt is compressed by default: retrieved schema chunks are compacted
        and the instruction template is intentionally short to reduce local model
        input tokens.
        """
        with self._generation_lock:
            schema_context = self.schema_context_provider.retrieve_schema_context(user_request)
            raw_response = self.llm_client.generate(self._build_prompt(schema_context, user_request))

        sample = parse_generated_sample(raw_response)
        validation_errors = validate_operation_against_schema(sample.operation, self.settings.schema_file)
        validation_errors.extend(validate_variable_usage(sample.operation, sample.variables))
        if validation_errors:
            raise RuntimeError(
                "Generated GraphQL operation was invalid for the current schema: "
                + "; ".join(validation_errors)
            )

        return sample

    def pre_warm(self) -> None:
        """Pre-load the local Ollama model during application startup.

        This is an inference optimization for the API path. It sends a tiny
        prompt through the configured LLM client so Ollama loads the model before
        the first user request. The setting trades a slightly slower startup for
        lower first-request latency.
        """
        if not self.settings.ollama_pre_warm_enabled:
            return

        self.llm_client.generate(self.settings.ollama_pre_warm_prompt)

    def _build_prompt(self, schema_context: str, user_request: str) -> str:
        user_prompt = GRAPHQL_USER_PROMPT_TEMPLATE.format(
            schema_context=schema_context,
            user_request=user_request,
        )
        return f"{GRAPHQL_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _build_default_llm_client(self) -> LLMClient:
        ollama_client = OllamaClient(settings=self.settings)
        if not self.settings.inference_cache_enabled:
            return ollama_client

        return CachedLLMClient(
            llm_client=ollama_client,
            cache=PromptResponseCache(self.settings.inference_cache_path),
            namespace=self.settings.inference_cache_namespace(),
        )


def parse_generated_sample(raw_response: str) -> GeneratedGraphQLSample:
    """Parse fenced model output into a structured sample query result."""
    code_blocks = re.findall(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", raw_response, flags=re.DOTALL)
    operation = code_blocks[0].strip() if code_blocks else raw_response.strip()
    variables: dict[str, Any] = {}

    if len(code_blocks) > 1:
        variables_text = code_blocks[1].strip()
        if variables_text:
            try:
                parsed_variables = json.loads(variables_text)
            except json.JSONDecodeError:
                parsed_variables = {"_raw": variables_text}

            if isinstance(parsed_variables, dict):
                variables = parsed_variables
            else:
                variables = {"value": parsed_variables}

    if not variables:
        variables = _infer_variables_from_operation(operation)

    return GeneratedGraphQLSample(
        operation=operation,
        variables=variables,
        raw_response=raw_response,
    )


def _infer_variables_from_operation(operation: str) -> dict[str, Any]:
    inferred_variables: dict[str, Any] = {}

    for variable_name, type_ref in VARIABLE_DEFINITION.findall(operation):
        inferred_variables[variable_name] = _sample_value_for_graphql_type(variable_name, type_ref)

    return inferred_variables


def _sample_value_for_graphql_type(variable_name: str, type_ref: str) -> Any:
    base_type = re.sub(r"[\[\]!]", "", type_ref)

    if type_ref.startswith("["):
        return [_sample_value_for_graphql_type(variable_name, base_type)]

    if base_type == "Boolean":
        return True
    if base_type == "Float":
        return 1.0
    if base_type == "Int":
        return 1
    if base_type == "ID":
        return "US" if "code" in variable_name.lower() else "example-id"

    return "example"


def validate_operation_against_schema(operation: str, schema_file: Any) -> list[str]:
    """Validate generated operation field selections against the local schema."""
    schema_fields = _parse_schema_field_types(schema_file.read_text(encoding="utf-8"))
    tokens = _tokenize_operation(operation)
    if not tokens:
        return ["operation was empty"]

    try:
        selection_start = tokens.index("{")
    except ValueError:
        return ["operation did not contain a selection set"]

    errors: list[str] = []
    _validate_selection_set(tokens, selection_start, "Query", schema_fields, errors)
    return errors


def validate_variable_usage(operation: str, variables: dict[str, Any]) -> list[str]:
    """Validate that returned variables are actually used by the GraphQL operation."""
    errors: list[str] = []
    for variable_name in variables:
        if variable_name.startswith("_"):
            continue
        if f"${variable_name}" not in operation:
            errors.append(f"variables JSON includes {variable_name}, but operation does not use ${variable_name}")

    return errors


def _parse_schema_field_types(schema_text: str) -> dict[str, dict[str, str]]:
    schema_fields: dict[str, dict[str, str]] = {}
    for type_match in re.finditer(r"\btype\s+([_A-Za-z][_0-9A-Za-z]*)\s*\{(.*?)\}", schema_text, re.DOTALL):
        type_name = type_match.group(1)
        body = type_match.group(2)
        fields: dict[str, str] = {}

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            field_match = re.match(
                r"([_A-Za-z][_0-9A-Za-z]*)\s*(?:\([^)]*\))?\s*:\s*([!\[\]_0-9A-Za-z]+)",
                line,
            )
            if field_match is not None:
                fields[field_match.group(1)] = field_match.group(2)

        schema_fields[type_name] = fields

    return schema_fields


def _tokenize_operation(operation: str) -> list[str]:
    cleaned_operation = re.sub(r'"(?:\\.|[^"\\])*"', '""', operation)
    cleaned_operation = re.sub(r"#.*", "", cleaned_operation)
    return re.findall(r"[_A-Za-z][_0-9A-Za-z]*|\{|\}|\(|\)|:|,", cleaned_operation)


def _validate_selection_set(
    tokens: list[str],
    start_index: int,
    parent_type: str,
    schema_fields: dict[str, dict[str, str]],
    errors: list[str],
) -> int:
    index = start_index + 1
    fields = schema_fields.get(parent_type, {})

    while index < len(tokens):
        token = tokens[index]
        if token == "}":
            return index + 1
        if token in {"{", "}", "(", ")", ":", ","}:
            index += 1
            continue

        field_name = token
        index += 1
        if index < len(tokens) - 1 and tokens[index] == ":":
            field_name = tokens[index + 1]
            index += 2

        if index < len(tokens) and tokens[index] == "(":
            index = _skip_balanced_tokens(tokens, index, "(", ")")

        field_type = fields.get(field_name)
        if field_type is None:
            errors.append(f"type {parent_type} has no field {field_name}")
            if index < len(tokens) and tokens[index] == "{":
                index = _skip_balanced_tokens(tokens, index, "{", "}")
            continue

        nested_type = _unwrap_graphql_type(field_type)
        if index < len(tokens) and tokens[index] == "{":
            if nested_type not in schema_fields:
                errors.append(f"scalar field {parent_type}.{field_name} must not have nested fields")
                index = _skip_balanced_tokens(tokens, index, "{", "}")
            else:
                index = _validate_selection_set(tokens, index, nested_type, schema_fields, errors)

    return index


def _skip_balanced_tokens(tokens: list[str], start_index: int, open_token: str, close_token: str) -> int:
    depth = 0
    index = start_index
    while index < len(tokens):
        if tokens[index] == open_token:
            depth += 1
        elif tokens[index] == close_token:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1

    return index


def _unwrap_graphql_type(type_ref: str) -> str:
    return re.sub(r"[\[\]!]", "", type_ref)
