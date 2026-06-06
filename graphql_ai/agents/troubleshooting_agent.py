from __future__ import annotations

import re
from threading import Lock
from typing import Any

from graphql_ai.core.config import AppSettings, get_settings
from graphql_ai.core.protocols import SchemaContextProvider
from graphql_ai.domain import TroubleshootingResult
from graphql_ai.llm.base import LLMClient
from graphql_ai.llm.factory import build_llm_client
from graphql_ai.rag.vector_store import SchemaVectorStore
from graphql_ai.services.sample_query_service import InvalidRootFieldNameError, validate_root_field_request


TROUBLESHOOTING_SYSTEM_PROMPT = (
    "You troubleshoot GraphQL operations using only the provided validation issues and schema context. "
    "Do not invent schema fields, arguments, types, or extra issues. Preserve submitted fields unless an "
    "issue explicitly says they are invalid. For syntax errors, fix only GraphQL structure. Return only "
    "DETAIL and SUGGESTION fenced code blocks. DETAIL must be 1 to 3 short explanation lines. SUGGESTION "
    "must be the full corrected GraphQL operation."
)

TROUBLESHOOTING_PROMPT_TEMPLATE = """DETAIL:
```text
1 to 3 short lines explaining the correction.
```

SUGGESTION:
```graphql
Full corrected GraphQL operation.
```

Correction rules:
- Use "Did you mean ..." from validation issues when available.
- Explain changes to the operation, not the schema.
- Do not remove valid submitted fields.
- Do not treat selected response fields as arguments.
- Do not replace a nested object field with a similar root Query field.

Root field:
{root_field}

Validation issues:
{issues}

Schema context:
{schema_context}

Submitted operation:
{graphql_call}
"""


CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", flags=re.DOTALL)


def validate_troubleshooting_input(root_field: str, graphql_call: str) -> tuple[str, str]:
    """Return normalized input or raise when the request is malformed."""
    normalized_root_field = validate_root_field_request(root_field)
    normalized_graphql_call = graphql_call.strip()
    if not normalized_graphql_call:
        raise InvalidRootFieldNameError("GraphQL call body must not be empty.")

    return normalized_root_field, normalized_graphql_call


class GraphQLValidator:
    """Validator that captures GraphQL syntax and schema issues.

    The GraphQL schema is parsed once when the tool is created because the
    local SDL file changes rarely and validation may run multiple times per
    troubleshooting request.
    """

    def __init__(self, schema_file: Any) -> None:
        """Create a validator with a cached parsed schema."""
        self.schema_file = schema_file
        self.schema = self._build_schema()

    def validate(self, graphql_call: str) -> list[str]:
        """Return GraphQL syntax and schema issues for a submitted operation.

        This deterministic validator runs before any model inference. Syntax errors
        come from GraphQL parsing, while schema errors come from validating the
        parsed document against the local SDL. Each issue keeps line and column
        details so the model can explain the correction in user-friendly terms.
        """
        try:
            from graphql import parse, validate
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

        try:
            document = parse(graphql_call)
        except Exception as exc:
            return [format_graphql_issue(exc)]

        return [format_graphql_issue(error) for error in validate(self.schema, document)]

    def _build_schema(self) -> Any:
        try:
            from graphql import build_schema
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

        return build_schema(self.schema_file.read_text(encoding="utf-8"))


class SchemaContextRetriever:
    """Retriever that caches RAG schema context for troubleshooting."""

    def __init__(self, schema_context_provider: SchemaContextProvider) -> None:
        """Create a retriever with an in-memory cache per root field."""
        self.schema_context_provider = schema_context_provider
        self._cache: dict[str, str] = {}

    def retrieve(self, root_field: str) -> str:
        """Retrieve RAG schema context for the root field being troubleshot.

        The troubleshooting prompt does not receive the whole schema. Instead,
        this retriever asks the configured schema-context provider for chunks related
        to the requested Query or Mutation field and caches that context for the
        lifetime of this instance.
        """
        if root_field not in self._cache:
            self._cache[root_field] = self.schema_context_provider.retrieve_schema_context(
                f"Troubleshoot GraphQL Query or Mutation root field {root_field}"
            )

        return self._cache[root_field]


class TroubleshootingAgent:
    """Service that troubleshoots user-provided GraphQL operations.

    The service runs deterministic input guardrails, validates the user's
    GraphQL operation, retrieves focused schema context with RAG, and asks the
    LLM only to explain the issue and propose a candidate fix. The suggested
    operation is validated before it is returned.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        schema_context_provider: SchemaContextProvider | None = None,
        allow_downloads: bool = False,
    ) -> None:
        """Create a troubleshooting service with injectable dependencies."""
        self.settings = settings or get_settings()
        self.llm_client = llm_client or self._build_default_llm_client()
        self.schema_context_provider = schema_context_provider or SchemaVectorStore(
            settings=self.settings,
            allow_downloads=allow_downloads,
        )
        self.validator = GraphQLValidator(self.settings.schema_file)
        self.schema_context_retriever = SchemaContextRetriever(self.schema_context_provider)
        self._inference_lock = Lock()

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        """Troubleshoot a user-submitted GraphQL operation.

        This is the main workflow behind the troubleshoot service:

        1. Validate and normalize the root-field path value and request body.
        2. Run GraphQL validation before calling the model.
        3. Return immediately when the submitted operation is already valid.
        4. Retrieve focused schema context for the requested root field.
        5. Build a prompt from validation issues, schema context, and the
           submitted operation.
        6. Ask the configured LLM provider for detail text and a candidate fix.
        7. Clean the model detail so API users see explanation, not raw errors.
        8. Validate the suggested operation before returning it.

        Deterministic validation output stays authoritative. The model can explain and
        propose a correction, but GraphQL-core validation decides whether the
        original call is invalid and whether the suggestion is safe to return.
        """
        normalized_root_field, normalized_graphql_call = validate_troubleshooting_input(root_field, graphql_call)
        validation_issues = self.validator.validate(normalized_graphql_call)
        if not validation_issues:
            return TroubleshootingResult(
                root_field=normalized_root_field,
                status="valid",
                issues=[],
                detail=[],
                suggestion="",
                raw_response="",
            )

        schema_context = self.schema_context_retriever.retrieve(normalized_root_field)

        with self._inference_lock:
            raw_response = self.llm_client.generate(
                self._build_prompt(
                    root_field=normalized_root_field,
                    graphql_call=normalized_graphql_call,
                    schema_context=schema_context,
                    issues=validation_issues,
                )
            )

        detail, suggested_operation = parse_troubleshooting_response(raw_response)
        detail = clean_model_detail(detail, validation_issues)

        corrected_issues = []
        if suggested_operation:
            corrected_issues = self.validator.validate(suggested_operation)
            if corrected_issues:
                suggested_operation = ""

        issues = validation_issues
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
            root_field=root_field,
            schema_context=schema_context,
            issues="\n".join(f"- {issue}" for issue in issues) if issues else "- No validation issues found.",
            graphql_call=graphql_call,
        )
        return f"{TROUBLESHOOTING_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _build_default_llm_client(self) -> LLMClient:
        return build_llm_client(self.settings, namespace_prefix="troubleshooting")


def parse_troubleshooting_response(raw_response: str) -> tuple[list[str], str]:
    """Parse inference into detail text and a suggested operation.

    The troubleshooting prompt asks for labeled `DETAIL` and `SUGGESTION` code
    blocks. This parser prefers those labels, but also supports simple fenced
    block fallback so the educational app remains tolerant of imperfect local
    model formatting.
    """
    labeled_detail = _extract_labeled_code_block(raw_response, "DETAIL")
    labeled_suggestion = _extract_labeled_code_block(raw_response, "SUGGESTION")
    if labeled_detail is not None or labeled_suggestion is not None:
        return normalize_detail(labeled_detail or ""), (labeled_suggestion or "").strip()

    code_blocks = CODE_BLOCK_RE.findall(raw_response)
    if not code_blocks:
        return normalize_detail(raw_response), ""
    if len(code_blocks) == 1 and looks_like_graphql_operation(code_blocks[0]):
        return [], code_blocks[0].strip()

    detail = normalize_detail(code_blocks[0])
    suggested_operation = code_blocks[1].strip() if len(code_blocks) > 1 else ""
    return detail, suggested_operation


def _extract_labeled_code_block(raw_response: str, label: str) -> str | None:
    pattern = rf"{label}\s*:\s*```(?:[A-Za-z0-9_-]+)?\s*(.*?)```"
    match = re.search(pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    return match.group(1)


def normalize_detail(raw_detail: str) -> list[str]:
    """Normalize model detail output into readable response lines.

    The API returns `detail` as a list of short lines. This helper removes empty
    lines and surrounding whitespace while preserving the model's wording.
    """
    detail = raw_detail.strip()
    if not detail:
        return []

    return [line.strip() for line in detail.splitlines() if line.strip()]


def clean_model_detail(detail: list[str], issues: list[str]) -> list[str]:
    """Keep only explanation lines for the response detail field.

    GraphQL-core issues are already returned in the `issues` field. This helper
    removes model lines that simply repeat those validator messages so users see
    a concise explanation of the correction instead of duplicate raw errors.
    """
    normalized_issues = [_normalize_issue_line(issue) for issue in issues]
    cleaned_detail = []

    for line in detail:
        cleaned_line = line.strip().lstrip("- ").strip()
        if not cleaned_line:
            continue

        normalized_line = _normalize_issue_line(cleaned_line)
        if any(issue in normalized_line or normalized_line in issue for issue in normalized_issues):
            continue

        cleaned_detail.append(cleaned_line)

    return cleaned_detail


def looks_like_graphql_operation(value: str) -> bool:
    """Return whether text appears to be a GraphQL operation."""
    stripped_value = value.lstrip()
    return stripped_value.startswith(("query ", "mutation ", "subscription ", "{"))


def _normalize_issue_line(value: str) -> str:
    normalized_value = value.strip().lstrip("- ").strip()
    normalized_value = re.sub(r"^\d+\.\s*", "", normalized_value)
    normalized_value = normalized_value.strip("*` ")
    return re.sub(r"\s+Location:.*$", "", normalized_value).strip()


def format_graphql_issue(error: Exception) -> str:
    """Format GraphQL parser or validator errors with line and column locations."""
    message = getattr(error, "message", str(error).split("\n\n", maxsplit=1)[0])
    locations = getattr(error, "locations", None) or []
    if not locations:
        return str(message)

    location_text = ", ".join(f"line {location.line}, column {location.column}" for location in locations)
    return f"{message} Location: {location_text}."
