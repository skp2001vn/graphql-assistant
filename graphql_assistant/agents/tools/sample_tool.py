from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

from graphql_assistant.core.config import AppSettings, get_settings
from graphql_assistant.core.protocols import SchemaContextProvider
from graphql_assistant.domain import GeneratedGraphQLSample
from graphql_assistant.llm.base import LLMClient
from graphql_assistant.llm.factory import build_llm_client
from graphql_assistant.llm.pre_warm import LLMPreWarmer
from graphql_assistant.rag.vector_store import SchemaVectorStore


GRAPHQL_SYSTEM_PROMPT = (
    "You are a GraphQL expert. Generate one valid GraphQL operation from the provided schema. "
    "Use only schema fields. The first field inside the operation body must be exactly the "
    "requested root field, including singular or plural spelling. Use variables for required "
    "arguments and put sample values only in the variables JSON. If the root field has no "
    "arguments, do not add arguments and return empty variables JSON. Include all fields defined "
    "on the selected response type. Expand nested object and list fields only when those fields "
    "exist on that type. Never add fields from another response type or inferred reverse "
    "relationships. Do not select Query or Mutation root fields inside response objects. "
    "Return exactly two fenced code blocks: GraphQL operation, then variables JSON."
)

GRAPHQL_PROMPT_TEMPLATE = """Schema:
{schema_context}

Root field:
{root_field}

Response type:
{response_type}

Operation name:
{operation_name}
"""

VARIABLE_DEFINITION = re.compile(r"\$([_A-Za-z][_0-9A-Za-z]*)\s*:\s*([!\[\]_0-9A-Za-z]+)")
GRAPHQL_NAME = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")


class InvalidRootFieldNameError(ValueError):
    """Raised when an API root field is not a valid GraphQL field name."""


class RootFieldNotInSchemaError(ValueError):
    """Raised when a requested root field is not defined on Query or Mutation."""


class SampleTool:
    """RAG-backed assistant tool for sample GraphQL operation generation.

    This tool owns the business workflow behind "generate a sample operation"
    requests. It turns a single GraphQL root-field request into a prompt-safe,
    schema-valid sample by combining deterministic validation with LLM
    inference:

    1. Validate and normalize the requested Query or Mutation root field.
    2. Retrieve focused schema context for that field through the app's RAG
       layer instead of sending the full SDL into every prompt.
    3. Build a constrained generation prompt that anchors the model on the
       requested root field, response type, operation name, and variable
       rules.
    4. Run inference through the configured LLM client.
    5. Parse the model response into a GraphQL operation and variables JSON.
    6. Apply post-generation guardrails with GraphQL-core validation and
       variable-usage checks before returning the sample.

    The design is intentionally hybrid: retrieval-augmented prompting improves
    relevance, while deterministic validation limits hallucinated fields and
    malformed variables from leaking into the final response.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        llm_pre_warmer: LLMPreWarmer | None = None,
        schema_context_provider: SchemaContextProvider | None = None,
        rebuild_index: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        """Create the sample-generation tool and wire its inference dependencies.

        The constructor supports dependency injection so tests can substitute
        fake LLM and RAG implementations, while production code can use the
        default vector-store retriever, configured model client, and optional
        model pre-warmer. The tool also keeps a small generation lock because
        some local-model runtimes behave more predictably under serialized
        prompt execution.
        """
        self.settings = settings or get_settings()
        self.schema_context_provider = schema_context_provider or SchemaVectorStore(
            settings=self.settings,
            rebuild=rebuild_index,
            allow_downloads=allow_downloads,
        )
        self.llm_client = llm_client or self._build_default_llm_client()
        self.llm_pre_warmer = llm_pre_warmer or LLMPreWarmer(self.settings, self.llm_client)
        self._generation_lock = Lock()

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        """Generate a schema-valid sample operation for the requested root field.

        Business flow:
        1. Normalize and validate the user-supplied root field.
        2. Retrieve topically relevant schema context through semantic search.
        3. Prompt the LLM to synthesize one GraphQL operation plus variables.
        4. Parse the raw model output into structured application data.
        5. Reject invalid generations with deterministic GraphQL validation.

        The returned value is intended to be directly consumable by the API or
        CLI. If the model proposes fields, arguments, or variables that do not
        match the current schema, this method fails fast instead of returning a
        partially-correct sample.
        """
        normalized_root_field = validate_root_field_against_schema(root_field, self.settings.schema_file)

        operation_name = f"{_pascal_case(normalized_root_field)}Query"
        response_type = _response_type_name(normalized_root_field)
        retrieval_request = f"GraphQL Query or Mutation root field {normalized_root_field}"

        with self._generation_lock:
            schema_context = self.schema_context_provider.retrieve_schema_context(retrieval_request)
            raw_response = self.llm_client.generate(
                self._build_prompt(
                    schema_context=schema_context,
                    root_field=normalized_root_field,
                    response_type=response_type,
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

    def _build_prompt(
        self,
        schema_context: str,
        root_field: str,
        response_type: str,
        operation_name: str,
    ) -> str:
        user_prompt = GRAPHQL_PROMPT_TEMPLATE.format(
            schema_context=schema_context,
            root_field=root_field,
            response_type=response_type,
            operation_name=operation_name,
        )
        return f"{GRAPHQL_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _build_default_llm_client(self) -> LLMClient:
        return build_llm_client(self.settings)


def parse_generated_sample(raw_response: str) -> GeneratedGraphQLSample:
    """Parse model output into a GraphQL operation and Variables JSON."""
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
    """Validate generated operations with GraphQL-core as an output guardrail."""
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
    """Validate that returned Variables JSON entries are used by the operation."""
    errors: list[str] = []
    for variable_name in variables:
        if variable_name.startswith("_"):
            continue
        if f"${variable_name}" not in operation:
            errors.append(f"variables JSON includes {variable_name}, but operation does not use ${variable_name}")

    return errors


def validate_root_field_request(root_field: str) -> str:
    """Validate an API root-field request before RAG retrieval and inference."""
    normalized_root_field = root_field.strip()
    if not normalized_root_field:
        raise InvalidRootFieldNameError("Root field must not be empty.")
    if GRAPHQL_NAME.fullmatch(normalized_root_field) is None:
        raise InvalidRootFieldNameError(
            "Root field must be a GraphQL field name, for example: country."
        )

    return normalized_root_field


def validate_root_field_against_schema(root_field: str, schema_file: Any) -> str:
    """Validate that a requested root field exists on Query or Mutation.

    This keeps assistant requests deterministic before retrieval or inference.
    The function first validates GraphQL field-name syntax, then checks the
    active schema's Query and Mutation root types for the requested field.
    """
    normalized_root_field = validate_root_field_request(root_field)
    root_fields = _load_schema_root_fields(schema_file)
    if normalized_root_field not in root_fields:
        available_fields = ", ".join(sorted(root_fields))
        raise RootFieldNotInSchemaError(
            "No GraphQL Query or Mutation field named "
            f"`{normalized_root_field}` exists in the current schema. "
            f"Available root fields: {available_fields}."
        )

    return normalized_root_field


def _load_schema_root_fields(schema_file: Any) -> set[str]:
    try:
        from graphql import build_schema
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

    schema = build_schema(schema_file.read_text(encoding="utf-8"))
    root_fields: set[str] = set()

    if schema.query_type is not None:
        root_fields.update(schema.query_type.fields.keys())
    if schema.mutation_type is not None:
        root_fields.update(schema.mutation_type.fields.keys())

    return root_fields


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-\s]+", value) if part)


def _response_type_name(root_field: str) -> str:
    if root_field.endswith("ies"):
        root_field = f"{root_field[:-3]}y"
    elif root_field.endswith("s"):
        root_field = root_field[:-1]

    return _pascal_case(root_field)


def _format_graphql_error(error: Exception) -> str:
    return str(error).split("\n\n", maxsplit=1)[0]
