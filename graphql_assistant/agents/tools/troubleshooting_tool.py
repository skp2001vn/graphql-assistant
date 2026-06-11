from __future__ import annotations

import re
from threading import Lock
from typing import Any

from graphql_assistant.agents.tools.sample_tool import InvalidRootFieldNameError, validate_root_field_request
from graphql_assistant.core.config import AppSettings, get_settings
from graphql_assistant.core.protocols import SchemaContextProvider
from graphql_assistant.domain import TroubleshootingResult
from graphql_assistant.llm.base import LLMClient
from graphql_assistant.llm.factory import build_llm_client
from graphql_assistant.llm.pre_warm import LLMPreWarmer
from graphql_assistant.rag.vector_store import SchemaVectorStore


TROUBLESHOOTING_SYSTEM_PROMPT = (
    "You troubleshoot GraphQL operations using only the provided validation issues and schema context. "
    "Your response must contain exactly two sections in this order: DETAIL, then SUGGESTION. "
    "Each section must be a fenced code block. DETAIL is required and must not be empty. "
    "SUGGESTION is required and must contain the full corrected GraphQL operation. Do not return JSON. "
    "Do not return bullets outside the fenced blocks. Do not add explanations outside the fenced blocks. "
    "Do not invent schema fields, arguments, types, or extra issues. Preserve submitted fields unless a "
    "validation issue explicitly says they are invalid. For syntax errors, fix only GraphQL structure."
)

TROUBLESHOOTING_PROMPT_TEMPLATE = """You must return exactly this format and nothing else:
DETAIL:
```text
A short explanation of the correction. This block is required and must not be empty.
```

SUGGESTION:
```graphql
Full corrected GraphQL operation.
```

DETAIL rules:
- Write 1 to 3 short natural-language lines.
- Explain the correction, not the schema.
- Do not copy the raw validation issue verbatim.
- Do not include GraphQL code in DETAIL.

SUGGESTION rules:
- Return the full corrected operation, not only the changed field.
- Use "Did you mean ..." from validation issues when available.
- Preserve valid submitted fields.
- For syntax errors, fix only GraphQL structure such as braces, parentheses, colons, commas, and variable syntax.
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
    """Normalize troubleshooting input before validation and inference.

    This is the first guardrail in the troubleshooting workflow. It keeps the
    downstream validator and LLM prompt focused on a legitimate root field and
    a non-empty submitted operation instead of trying to recover from malformed
    request payloads later in the pipeline.
    """
    normalized_root_field = validate_root_field_request(root_field)
    normalized_graphql_call = graphql_call.strip()
    if not normalized_graphql_call:
        raise InvalidRootFieldNameError("GraphQL call body must not be empty.")

    return normalized_root_field, normalized_graphql_call


class GraphQLValidator:
    """Deterministic GraphQL-core validator for troubleshooting requests.

    The assistant uses this component before any LLM call. It separates
    parser/schema validation from model reasoning so the tool can:

    - short-circuit valid operations without spending tokens,
    - feed concrete validation issues into the prompt as grounded evidence,
    - and verify whether the model's proposed correction is actually valid.
    """

    def __init__(self, schema_file: Any) -> None:
        """Create a validator with a parsed schema cached in memory.

        The schema is built once at construction time because troubleshooting
        requests may validate both the original operation and a model-proposed
        correction in the same tool run.
        """
        self.schema_file = schema_file
        self.schema = self._build_schema()

    def validate(self, graphql_call: str) -> list[str]:
        """Return parser and schema-validation issues for a submitted operation.

        The output is a normalized list of human-readable issues that can be
        reused both as API-facing diagnostics and as grounded prompt context
        for corrective generation.
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
    """Lightweight schema-context retriever for troubleshooting prompts.

    Troubleshooting and sample generation both use retrieval-augmented
    prompting, but the retrieval query here is optimized for error analysis
    rather than synthesis. The retriever also caches context per root field so
    repeated troubleshooting calls avoid redundant vector lookups.
    """

    def __init__(self, schema_context_provider: SchemaContextProvider) -> None:
        """Create a retriever with in-memory caching keyed by root field."""
        self.schema_context_provider = schema_context_provider
        self._cache: dict[str, str] = {}

    def retrieve(self, root_field: str) -> str:
        """Retrieve prompt-ready schema context for the field under analysis.

        The returned text is not the entire schema. It is the narrowed context
        retrieved from the RAG layer and injected into the troubleshooting
        prompt to reduce prompt size while preserving relevant type and field
        definitions.
        """
        if root_field not in self._cache:
            self._cache[root_field] = self.schema_context_provider.retrieve_schema_context(
                f"Troubleshoot GraphQL Query or Mutation root field {root_field}"
            )

        return self._cache[root_field]


class TroubleshootingTool:
    """Assistant tool for GraphQL validation and corrective suggestion.

    This tool owns the "troubleshoot a GraphQL operation" business workflow.
    It combines deterministic GraphQL validation with targeted LLM reasoning:

    1. Normalize the submitted root field and operation text.
    2. Validate the operation against the current schema.
    3. Return immediately when the operation is already valid.
    4. Retrieve focused schema context through the RAG layer.
    5. Prompt the LLM with validation issues, schema context, and the original
       operation to produce an explanation plus a corrected candidate.
    6. Re-validate the suggested correction before returning it.

    This keeps the model in a constrained correction role instead of asking it
    to infer schema truth from scratch. The validation-repair-validation loop
    is the main hallucination-control technique in this workflow.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        llm_pre_warmer: LLMPreWarmer | None = None,
        schema_context_provider: SchemaContextProvider | None = None,
        allow_downloads: bool = False,
    ) -> None:
        """Create the troubleshooting tool with injectable validator/RAG/LLM dependencies.

        The default wiring uses the configured schema file, vector-store-backed
        schema retrieval, and the troubleshooting LLM namespace. Tests can swap
        in fakes for deterministic behavior. The tool keeps its own inference
        lock because corrective prompting is usually served by the same local
        model runtime as generation.
        """
        self.settings = settings or get_settings()
        self.llm_client = llm_client or self._build_default_llm_client()
        self.llm_pre_warmer = llm_pre_warmer or LLMPreWarmer(self.settings, self.llm_client)
        self.schema_context_provider = schema_context_provider or SchemaVectorStore(
            settings=self.settings,
            allow_downloads=allow_downloads,
        )
        self.validator = GraphQLValidator(self.settings.schema_file)
        self.schema_context_retriever = SchemaContextRetriever(self.schema_context_provider)
        self._inference_lock = Lock()

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        """Analyze a submitted operation and return validation-aware guidance.

        The method first runs deterministic validation. If the submitted
        operation is already valid, it returns a `valid` result without calling
        the LLM. Otherwise it performs retrieval-augmented corrective
        generation, parses the model response into explanation/detail and a
        suggested operation, then re-validates that suggestion before exposing
        it to the caller.

        This two-pass validation strategy is the key business rule: the tool
        can explain model output, but it does not trust model output until the
        schema validator accepts it.
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
            self.llm_pre_warmer.pre_warm()
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
    """Parse inference into detail text and a suggested operation."""
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
    """Normalize model detail output into readable response lines."""
    detail = raw_detail.strip()
    if not detail:
        return []

    return [line.strip() for line in detail.splitlines() if line.strip()]


def clean_model_detail(detail: list[str], issues: list[str]) -> list[str]:
    """Keep only explanation lines for the response detail field."""
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
