from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Protocol

from graphql_assistant.agents.tools import (
    SampleTool,
    TroubleshootingTool,
    validate_operation_against_schema,
    validate_variable_usage,
)
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult


@dataclass(frozen=True)
class SamplePromptEvalCase:
    """Prompt evaluation case for sample GraphQL generation."""

    name: str
    root_field: str
    expected_text: tuple[str, ...]


@dataclass(frozen=True)
class TroubleshootingPromptEvalCase:
    """Prompt evaluation case for GraphQL troubleshooting."""

    name: str
    root_field: str
    graphql_call: str
    expected_suggestion_text: tuple[str, ...]


@dataclass(frozen=True)
class PromptEvalResult:
    """Prompt evaluation result with simple pass/fail checks."""

    workflow: str
    name: str
    passed: bool
    checks: tuple[str, ...]
    error: str = ""


class SettingsWithSchemaFile(Protocol):
    """Protocol for settings objects that expose the active GraphQL schema file."""

    schema_file: object


class SampleGenerationTool(Protocol):
    """Protocol for tools that generate sample GraphQL operations for evals."""

    settings: SettingsWithSchemaFile

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        """Generate a sample GraphQL operation for a root field."""


class TroubleshootingRunner(Protocol):
    """Protocol for tools that troubleshoot GraphQL operations for evals."""

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        """Troubleshoot a GraphQL operation."""


DEFAULT_SAMPLE_CASES = (
    SamplePromptEvalCase(
        name="sample country by code",
        root_field="country",
        expected_text=("country(code:", "code", "name"),
    ),
    SamplePromptEvalCase(
        name="sample countries list",
        root_field="countries",
        expected_text=("countries", "code", "name"),
    ),
)

DEFAULT_TROUBLESHOOTING_CASES = (
    TroubleshootingPromptEvalCase(
        name="fix selected field typo",
        root_field="country",
        graphql_call="""
query CountryQuery($code: ID!) {
  country(code: $code) {
    code1
    name
  }
}
""",
        expected_suggestion_text=("country(code:", "code", "name"),
    ),
)


def run_sample_prompt_eval_cases(
    tool: SampleGenerationTool,
    cases: Iterable[SamplePromptEvalCase] = DEFAULT_SAMPLE_CASES,
) -> list[PromptEvalResult]:
    """Run sample-generation prompt evaluation cases.

    Each case calls the real sample-generation workflow, then scores the output
    with the same guardrails used by the API: GraphQL schema validation,
    variable-usage validation, and a few simple expected-text checks. This keeps
    prompt evaluation educational and deterministic without adding a separate
    eval framework.
    """
    results = []
    schema_file = tool.settings.schema_file

    for case in cases:
        try:
            sample = tool.generate(case.root_field)
            checks = _score_sample(case, sample, schema_file)
            results.append(PromptEvalResult("sample", case.name, _all_checks_passed(checks), checks))
        except Exception as exc:
            results.append(PromptEvalResult("sample", case.name, False, (), str(exc)))

    return results


def run_troubleshooting_prompt_eval_cases(
    tool: TroubleshootingRunner,
    schema_file: object,
    cases: Iterable[TroubleshootingPromptEvalCase] = DEFAULT_TROUBLESHOOTING_CASES,
) -> list[PromptEvalResult]:
    """Run troubleshooting prompt evaluation cases.

    Each case submits an intentionally invalid GraphQL operation to the
    troubleshooting tool, then checks that the tool reports issues, produces
    user-facing detail text, returns a suggestion, and that the suggestion
    validates against the schema.
    """
    results = []

    for case in cases:
        try:
            result = tool.troubleshoot(case.root_field, case.graphql_call)
            checks = _score_troubleshooting(case, result, schema_file)
            results.append(PromptEvalResult("troubleshoot", case.name, _all_checks_passed(checks), checks))
        except Exception as exc:
            results.append(PromptEvalResult("troubleshoot", case.name, False, (), str(exc)))

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt evaluation."""
    parser = argparse.ArgumentParser(description="Run simple prompt evaluation cases.")
    parser.add_argument(
        "--workflow",
        choices=("all", "sample", "troubleshoot"),
        default="all",
        help="Prompt workflow to evaluate.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the Chroma schema index before evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run prompt evaluation cases against the configured LLM provider."""
    args = parse_args()
    sample_tool = SampleTool(rebuild_index=args.rebuild)
    sample_tool.llm_pre_warmer.pre_warm()
    results: list[PromptEvalResult] = []

    if args.workflow in {"all", "sample"}:
        results.extend(run_sample_prompt_eval_cases(sample_tool))

    if args.workflow in {"all", "troubleshoot"}:
        troubleshooting_tool = TroubleshootingTool(
            settings=sample_tool.settings,
            llm_client=sample_tool.llm_client,
            llm_pre_warmer=sample_tool.llm_pre_warmer,
            schema_context_provider=sample_tool.schema_context_provider,
        )
        results.extend(
            run_troubleshooting_prompt_eval_cases(
                troubleshooting_tool,
                sample_tool.settings.schema_file,
            )
        )

    _print_results(results)
    if any(not result.passed for result in results):
        raise SystemExit(1)


def _score_sample(case: SamplePromptEvalCase, sample: GeneratedGraphQLSample, schema_file: object) -> tuple[str, ...]:
    checks = []
    validation_errors = validate_operation_against_schema(sample.operation, schema_file)
    variable_errors = validate_variable_usage(sample.operation, sample.variables)
    checks.append(_format_check("operation validates against schema", not validation_errors, validation_errors))
    checks.append(_format_check("variables match operation", not variable_errors, variable_errors))

    for expected_text in case.expected_text:
        checks.append(
            _format_check(
                f"operation contains `{expected_text}`",
                expected_text in sample.operation,
            )
        )

    return tuple(checks)


def _score_troubleshooting(
    case: TroubleshootingPromptEvalCase,
    result: TroubleshootingResult,
    schema_file: object,
) -> tuple[str, ...]:
    checks = [
        _format_check("original call reports validation issues", bool(result.issues)),
        _format_check("detail explains the correction", bool(result.detail)),
        _format_check("suggestion is returned", bool(result.suggestion.strip())),
    ]

    if result.suggestion:
        validation_errors = validate_operation_against_schema(result.suggestion, schema_file)
        checks.append(_format_check("suggestion validates against schema", not validation_errors, validation_errors))
    else:
        checks.append(_format_check("suggestion validates against schema", False, ["missing suggestion"]))

    for expected_text in case.expected_suggestion_text:
        checks.append(
            _format_check(
                f"suggestion contains `{expected_text}`",
                expected_text in result.suggestion,
            )
        )

    return tuple(checks)


def _format_check(name: str, passed: bool, details: list[str] | None = None) -> str:
    status = "PASS" if passed else "FAIL"
    if details:
        return f"{status} {name}: {'; '.join(details)}"

    return f"{status} {name}"


def _all_checks_passed(checks: tuple[str, ...]) -> bool:
    return all(check.startswith("PASS ") for check in checks)


def _print_results(results: list[PromptEvalResult]) -> None:
    print("\nPrompt evaluation results:\n")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.workflow}: {result.name}")
        if result.error:
            print(f"  - ERROR {result.error}")
        for check in result.checks:
            print(f"  - {check}")

    passed_count = sum(1 for result in results if result.passed)
    print(f"\nSummary: {passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()
