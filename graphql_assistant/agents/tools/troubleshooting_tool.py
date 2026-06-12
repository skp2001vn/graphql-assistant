from __future__ import annotations

import re
from threading import Lock
from typing import Any

from graphql_assistant.agents.tools.sample_tool import (
    InvalidRootFieldNameError,
    validate_root_field_against_schema,
)
from graphql_assistant.core.config import AppSettings, get_settings
from graphql_assistant.core.protocols import SchemaContextProvider
from graphql_assistant.domain import TroubleshootingResult
from graphql_assistant.llm.base import LLMClient
from graphql_assistant.llm.factory import build_llm_client
from graphql_assistant.llm.pre_warm import LLMPreWarmer
from graphql_assistant.rag.vector_store import SchemaVectorStore


TROUBLESHOOTING_SYSTEM_PROMPT = (
    "You are a GraphQL expert. You fix GraphQL operations using only the provided validation issues and schema context. "
    "Return exactly two fenced sections: DETAIL with 1-3 short correction lines, then SUGGESTION "
    "with the full corrected GraphQL operation. Do not return JSON or text outside those sections. "
    "Preserve valid submitted fields. Do not invent schema fields, arguments, types, or extra issues. "
    "For syntax errors, fix only GraphQL structure."
)

TROUBLESHOOTING_PROMPT_TEMPLATE = """Return format:
DETAIL:
```text
Short correction detail.
```

SUGGESTION:
```graphql
Full corrected GraphQL operation.
```

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


def validate_troubleshooting_input(root_field: str, graphql_call: str, schema_file: Any) -> tuple[str, str]:
    """Normalize troubleshooting input before validation and inference.

    This is the first guardrail in the troubleshooting workflow. It keeps the
    downstream validator and LLM prompt focused on a legitimate root field and
    a non-empty submitted operation instead of trying to recover from malformed
    request payloads later in the pipeline.
    """
    normalized_root_field = validate_root_field_against_schema(root_field, schema_file)
    normalized_graphql_call = graphql_call.strip()
    if not normalized_graphql_call:
        raise InvalidRootFieldNameError("GraphQL call body must not be empty.")

    return normalized_root_field, normalized_graphql_call


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
        self._schema = self._build_schema()
        self._schema_context_cache: dict[str, str] = {}
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
        normalized_root_field, normalized_graphql_call = validate_troubleshooting_input(
            root_field,
            graphql_call,
            self.settings.schema_file,
        )
        validation_issues = self._validate_graphql(normalized_graphql_call)
        if not validation_issues:
            return TroubleshootingResult(
                root_field=normalized_root_field,
                status="valid",
                issues=[],
                detail=[],
                suggestion="",
                raw_response="",
            )

        schema_context = self._retrieve_schema_context(normalized_root_field)

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

        detail, suggested_operation = parse_troubleshooting_response(raw_response, validation_issues)

        corrected_issues = []
        if suggested_operation:
            corrected_issues = self._validate_graphql(suggested_operation)
            if corrected_issues:
                suggested_operation = ""
            else:
                suggested_operation = _format_graphql_operation(suggested_operation)

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

    def _retrieve_schema_context(self, root_field: str) -> str:
        if root_field not in self._schema_context_cache:
            self._schema_context_cache[root_field] = self.schema_context_provider.retrieve_schema_context(
                f"Troubleshoot GraphQL Query or Mutation root field {root_field}"
            )

        return self._schema_context_cache[root_field]

    def _validate_graphql(self, graphql_call: str) -> list[str]:
        try:
            from graphql import parse, validate
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

        try:
            document = parse(graphql_call)
        except Exception as exc:
            return [format_graphql_issue(exc)]

        return [format_graphql_issue(error) for error in validate(self._schema, document)]

    def _build_schema(self) -> Any:
        try:
            from graphql import build_schema
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

        return build_schema(self.settings.schema_file.read_text(encoding="utf-8"))


def parse_troubleshooting_response(raw_response: str, issues: list[str]) -> tuple[list[str], str]:
    """Parse inference into detail text and a suggested operation."""
    labeled_detail = _extract_labeled_code_block(raw_response, "DETAIL")
    labeled_suggestion = _extract_labeled_code_block(raw_response, "SUGGESTION")
    if labeled_detail is not None or labeled_suggestion is not None:
        return _parse_detail(labeled_detail or "", issues), (labeled_suggestion or "").strip()

    code_blocks = CODE_BLOCK_RE.findall(raw_response)
    if not code_blocks:
        return _parse_detail(raw_response, issues), ""
    if len(code_blocks) == 1 and looks_like_graphql_operation(code_blocks[0]):
        return [], code_blocks[0].strip()

    detail = _parse_detail(code_blocks[0], issues)
    suggested_operation = code_blocks[1].strip() if len(code_blocks) > 1 else ""
    return detail, suggested_operation


def _extract_labeled_code_block(raw_response: str, label: str) -> str | None:
    pattern = rf"{label}\s*:\s*```(?:[A-Za-z0-9_-]+)?\s*(.*?)```"
    match = re.search(pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    return match.group(1)


def _parse_detail(raw_detail: str, issues: list[str]) -> list[str]:
    """Normalize model detail output and keep only explanation lines."""
    detail = raw_detail.strip()
    if not detail:
        return []

    normalized_issues = [_normalize_issue_line(issue) for issue in issues]
    cleaned_detail = []

    for line in detail.splitlines():
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


def _format_graphql_operation(operation: str) -> str:
    """Render a validated GraphQL operation with stable pretty formatting."""
    try:
        from graphql import parse, print_ast
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

    return print_ast(parse(operation)).strip()
