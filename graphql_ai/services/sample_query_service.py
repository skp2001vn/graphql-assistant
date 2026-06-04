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
    "You are a GraphQL expert. Generate one valid GraphQL operation from the provided schema. "
    "Use only schema fields. Use the requested root field exactly. Use variables for required "
    "arguments and put sample values only in the variables JSON. If the root field has no "
    "arguments, do not add arguments and return empty variables JSON. Include all fields defined "
    "on the selected response type. Expand nested object and list fields only when those fields "
    "exist on that type. Never add fields from another response type or inferred reverse "
    "relationships. "
    "Return exactly two fenced code blocks: GraphQL operation, then variables JSON."
)

GRAPHQL_PROMPT_TEMPLATE = """Schema:
{schema_context}

Root field:
{root_field}

Operation name:
{operation_name}
"""

VARIABLE_DEFINITION = re.compile(r"\$([_A-Za-z][_0-9A-Za-z]*)\s*:\s*([!\[\]_0-9A-Za-z]+)")


class SampleQueryService:
    """Business service for generating sample GraphQL queries.

    This service coordinates the application workflow for the sample-query use case:
    it receives a root field name from the API, converts it into a short
    retrieval request, runs RAG retrieval through the configured schema-context
    provider, builds a prompt from retrieved context, sends that prompt through
    the configured LLM client for inference, and parses the model output into a
    GraphQL operation plus variables. In this application, the API `root_field`
    value is the schema Query or Mutation field name the user wants to generate,
    such as `country`.

    The current default schema-context provider is RAG-backed: `SchemaVectorStore`
    chunks the local GraphQL SDL, creates embeddings, stores them in a Chroma
    vector store, and retrieves relevant schema context for each request. The
    service depends on the `SchemaContextProvider` protocol, so that RAG can
    later be replaced or composed with other approaches such as agent workflows,
    planning, model routing, prompt evaluation, or inference optimization without
    changing the API layer.

    This service also applies GraphQL validation as a guardrail: model output is
    parsed and validated against the local SDL before the API returns it, which
    prevents invented fields or malformed operations from silently reaching
    callers.
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
        self._pre_warmed = False

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        """Generate a sample GraphQL operation and variables for an API root field.

        In this app, `root_field` means the schema Query or Mutation field name
        requested by the API, for example `country`. It is converted into a
        short retrieval request, then the full application flow runs: RAG
        retrieval, prompt construction, inference, parsing, and guardrail
        validation. The prompt is compressed by default: retrieved schema chunks
        are compacted and the instruction template is intentionally short to
        reduce local model input tokens. After generation, GraphQL-core
        validation rejects operations that do not match the current schema.
        """
        normalized_root_field = root_field.strip()
        if not normalized_root_field:
            raise ValueError("Root field must not be empty.")

        operation_name = f"{_pascal_case(normalized_root_field)}Query"
        retrieval_request = f"GraphQL Query or Mutation root field {normalized_root_field}"

        with self._generation_lock:
            self.pre_warm()
            schema_context = self.schema_context_provider.retrieve_schema_context(retrieval_request)
            raw_response = self.llm_client.generate(
                self._build_prompt(
                    schema_context=schema_context,
                    root_field=normalized_root_field,
                    operation_name=operation_name,
                )
            )

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
        """Pre-load the local Ollama model before custom inference.

        This inference optimization sends a tiny prompt through the configured
        LLM client so Ollama loads the model before the first custom generation.
        The setting trades a slightly slower first AI request for lower latency
        on following AI requests.
        """
        if self._pre_warmed:
            return

        if not self.settings.ollama_pre_warm_enabled:
            self._pre_warmed = True
            return

        self.llm_client.generate(self.settings.ollama_pre_warm_prompt)
        self._pre_warmed = True

    def _build_prompt(self, schema_context: str, root_field: str, operation_name: str) -> str:
        user_prompt = GRAPHQL_PROMPT_TEMPLATE.format(
            schema_context=schema_context,
            root_field=root_field,
            operation_name=operation_name,
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
    """Validate generated operations with GraphQL-core as an output guardrail.

    This guardrail uses the standard GraphQL parser and validator instead of
    custom string checks. It catches malformed GraphQL, invented fields, missing
    required arguments, invalid scalar selections, and variable type mismatches
    before a generated sample is returned.
    """
    try:
        from graphql import build_schema, parse, validate
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

    try:
        schema = build_schema(schema_file.read_text(encoding="utf-8"))
        document = parse(operation)
    except Exception as exc:
        return [_format_graphql_error(exc)]

    return [_format_graphql_error(error) for error in validate(schema, document)]


def validate_variable_usage(operation: str, variables: dict[str, Any]) -> list[str]:
    """Guardrail that validates returned variables are used by the operation."""
    errors: list[str] = []
    for variable_name in variables:
        if variable_name.startswith("_"):
            continue
        if f"${variable_name}" not in operation:
            errors.append(f"variables JSON includes {variable_name}, but operation does not use ${variable_name}")

    return errors


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-\s]+", value) if part)


def _format_graphql_error(error: Exception) -> str:
    return str(error).split("\n\n", maxsplit=1)[0]
