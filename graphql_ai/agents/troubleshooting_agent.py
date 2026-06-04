from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

from graphql_ai.core.config import AppSettings, get_settings
from graphql_ai.core.protocols import SchemaContextProvider
from graphql_ai.domain import TroubleshootingResult
from graphql_ai.llm.base import LLMClient
from graphql_ai.llm.cache import CachedLLMClient, PromptResponseCache
from graphql_ai.llm.ollama_client import OllamaClient
from graphql_ai.rag.vector_store import SchemaVectorStore
from graphql_ai.services.sample_query_service import InvalidRootFieldNameError, validate_root_field_request


TROUBLESHOOTING_PLAN = [
    "Validate input",
    "Parse and validate GraphQL",
    "Retrieve schema context",
    "Generate troubleshooting guidance",
    "Validate corrected operation",
]

TROUBLESHOOTING_SYSTEM_PROMPT = (
    "You are a GraphQL troubleshooting agent. Your goal is to explain what is wrong "
    "with a user's GraphQL operation and suggest a corrected operation. Use the tool "
    "observations and schema context. Tool observations are authoritative: do not add "
    "issues that are not listed in the validation issues. Do not invent schema fields. "
    "Return exactly two fenced code blocks: detail text, then a GraphQL suggested operation."
)

TROUBLESHOOTING_PROMPT_TEMPLATE = """Plan:
{plan}

Root field:
{root_field}

Schema context:
{schema_context}

Validation issues:
{issues}

User GraphQL operation:
{graphql_call}
"""


@dataclass(frozen=True)
class ValidationObservation:
    """Tool observation from parsing and validating a GraphQL operation."""

    issues: list[str]


class InputGuardrailTool:
    """Tool that validates root-field and body input before agent planning continues."""

    def validate(self, root_field: str, graphql_call: str) -> tuple[str, str]:
        """Return normalized input or raise when the request is malformed."""
        normalized_root_field = validate_root_field_request(root_field)
        normalized_graphql_call = graphql_call.strip()
        if not normalized_graphql_call:
            raise InvalidRootFieldNameError("GraphQL call body must not be empty.")

        return normalized_root_field, normalized_graphql_call


class GraphQLValidationTool:
    """Tool that captures GraphQL syntax and schema validation issues."""

    def __init__(self, schema_file: Any) -> None:
        """Create a validation tool for a local GraphQL SDL file."""
        self.schema_file = schema_file

    def validate(self, graphql_call: str) -> ValidationObservation:
        """Parse and validate a GraphQL operation, preserving line and column details."""
        try:
            from graphql import build_schema, parse, validate
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

        try:
            document = parse(graphql_call)
        except Exception as exc:
            return ValidationObservation([format_graphql_issue(exc)])

        schema = build_schema(self.schema_file.read_text(encoding="utf-8"))
        return ValidationObservation([format_graphql_issue(error) for error in validate(schema, document)])


class SchemaRetrievalTool:
    """Tool that retrieves RAG schema context for troubleshooting."""

    def __init__(self, schema_context_provider: SchemaContextProvider) -> None:
        """Create a retrieval tool from the configured schema-context provider."""
        self.schema_context_provider = schema_context_provider

    def retrieve(self, root_field: str) -> str:
        """Retrieve schema context for the root field being troubleshot."""
        return self.schema_context_provider.retrieve_schema_context(
            f"Troubleshoot GraphQL Query or Mutation root field {root_field}"
        )


class TroubleshootingAgent:
    """Tool-using agent for troubleshooting user-provided GraphQL operations.

    The agent has a goal, an explicit plan, and a small set of deterministic
    tools. It first runs input guardrails, then parses and validates the user's
    GraphQL operation, retrieves schema context with RAG, and finally calls the
    LLM for inference. The LLM suggestion is treated as a candidate answer:
    the corrected operation is validated before it is returned.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        schema_context_provider: SchemaContextProvider | None = None,
        allow_downloads: bool = False,
    ) -> None:
        """Create a troubleshooting agent with injectable tools and inference dependencies."""
        self.settings = settings or get_settings()
        self.llm_client = llm_client or self._build_default_llm_client()
        self.schema_context_provider = schema_context_provider or SchemaVectorStore(
            settings=self.settings,
            allow_downloads=allow_downloads,
        )
        self.input_tool = InputGuardrailTool()
        self.validation_tool = GraphQLValidationTool(self.settings.schema_file)
        self.retrieval_tool = SchemaRetrievalTool(self.schema_context_provider)
        self._inference_lock = Lock()

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        """Run the agent plan and return issues, detail, and suggested operation."""
        normalized_root_field, normalized_graphql_call = self.input_tool.validate(root_field, graphql_call)
        validation_observation = self.validation_tool.validate(normalized_graphql_call)
        if not validation_observation.issues:
            return TroubleshootingResult(
                root_field=normalized_root_field,
                status="valid",
                issues=[],
                detail="",
                suggestion="",
                raw_response="",
            )

        schema_context = self.retrieval_tool.retrieve(normalized_root_field)

        with self._inference_lock:
            raw_response = self.llm_client.generate(
                self._build_prompt(
                    root_field=normalized_root_field,
                    graphql_call=normalized_graphql_call,
                    schema_context=schema_context,
                    issues=validation_observation.issues,
                )
            )

        detail, suggested_operation = parse_troubleshooting_response(raw_response)
        if not suggested_operation and looks_like_graphql_operation(detail):
            suggested_operation = detail
            detail = default_detail_from_issues(validation_observation.issues)

        corrected_issues = []
        if suggested_operation:
            corrected_issues = self.validation_tool.validate(suggested_operation).issues
            if corrected_issues:
                suggested_operation = ""

        issues = validation_observation.issues
        if corrected_issues:
            issues = issues + [f"Corrected operation was still invalid: {issue}" for issue in corrected_issues]

        return TroubleshootingResult(
            root_field=normalized_root_field,
            status="valid" if not issues else "invalid",
            issues=issues,
            detail=detail,
            suggestion=suggested_operation,
            raw_response=raw_response,
        )

    def _build_prompt(
        self,
        root_field: str,
        graphql_call: str,
        schema_context: str,
        issues: list[str],
    ) -> str:
        user_prompt = TROUBLESHOOTING_PROMPT_TEMPLATE.format(
            plan="\n".join(f"- {step}" for step in TROUBLESHOOTING_PLAN),
            root_field=root_field,
            schema_context=schema_context,
            issues="\n".join(f"- {issue}" for issue in issues) if issues else "- No validation issues found.",
            graphql_call=graphql_call,
        )
        return f"{TROUBLESHOOTING_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _build_default_llm_client(self) -> LLMClient:
        ollama_client = OllamaClient(settings=self.settings)
        if not self.settings.inference_cache_enabled:
            return ollama_client

        return CachedLLMClient(
            llm_client=ollama_client,
            cache=PromptResponseCache(self.settings.inference_cache_path),
            namespace=f"troubleshooting|{self.settings.inference_cache_namespace()}",
        )


def parse_troubleshooting_response(raw_response: str) -> tuple[str, str]:
    """Parse agent inference into detail text and a suggested operation."""
    code_blocks = re.findall(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", raw_response, flags=re.DOTALL)
    if not code_blocks:
        return raw_response.strip(), ""

    detail = code_blocks[0].strip()
    suggested_operation = code_blocks[1].strip() if len(code_blocks) > 1 else ""
    return detail, suggested_operation


def looks_like_graphql_operation(value: str) -> bool:
    """Return whether text appears to be a GraphQL operation."""
    stripped_value = value.lstrip()
    return stripped_value.startswith(("query ", "mutation ", "subscription ", "{"))


def default_detail_from_issues(issues: list[str]) -> str:
    """Create deterministic detail text when inference only returns an operation."""
    if issues:
        return issues[0]

    return "No validation issues were found."


def format_graphql_issue(error: Exception) -> str:
    """Format GraphQL parser or validator errors with line and column locations."""
    message = getattr(error, "message", str(error).split("\n\n", maxsplit=1)[0])
    locations = getattr(error, "locations", None) or []
    if not locations:
        return str(message)

    location_text = ", ".join(f"line {location.line}, column {location.column}" for location in locations)
    return f"{message} Location: {location_text}."
