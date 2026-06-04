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
    "response type, expand nested objects/lists, use variables, and return exactly two fenced "
    "code blocks: GraphQL operation, then variables JSON."
)

GRAPHQL_USER_PROMPT_TEMPLATE = "Schema:\n{schema_context}\n\nRequest:\n{user_request}"

VARIABLE_DEFINITION = re.compile(r"\$([_A-Za-z][_0-9A-Za-z]*)\s*:\s*([!\[\]_0-9A-Za-z]+)")


def build_default_sample_request(target: str) -> str:
    """Build the default natural-language request for a sample GraphQL target."""
    normalized_target = target.strip().replace("-", " ")
    if not normalized_target:
        raise ValueError("Target must not be empty.")

    if normalized_target.lower() == "country":
        return (
            "Generate a sample GraphQL query named CountryQuery for a country by code. "
            "Use a variable named code with type ID. "
            "Include all available Country fields, including continent and languages. "
            "Return Variables JSON with code set to US."
        )

    return f"Generate a sample query for {normalized_target}"


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

        return parse_generated_sample(raw_response)

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
