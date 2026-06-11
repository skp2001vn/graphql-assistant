from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from graphql_assistant.agents import (
    AgentPlanningError,
    GraphQLAssistantAgent,
    GraphQLAssistantGoal,
    GraphQLAssistantResult,
)
from graphql_assistant.agents.tools import (
    SampleTool,
    TroubleshootingTool,
    validate_operation_against_schema,
    validate_variable_usage,
)
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult


AssistantEvalIntent = Literal["generate_sample", "troubleshoot", "unsupported"]


@dataclass(frozen=True)
class AssistantPromptEvalCase:
    """End-to-end evaluation case for the assistant API contract.

    The app is now centered on the `/assistant` surface, so prompt eval should
    primarily measure what that surface does: interpret a natural-language
    goal, route to the right workflow, and return a valid generation or repair
    result. These cases intentionally sit one layer above the individual tools.
    """

    name: str
    goal: str
    root_field: str
    expected_intent: AssistantEvalIntent
    expected_text: tuple[str, ...] = ()
    graphql_call: str | None = None
    expected_error_text: str = ""


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


class AssistantRunner(Protocol):
    """Protocol for the public assistant entrypoint used by evals."""

    def run(self, goal: GraphQLAssistantGoal) -> GraphQLAssistantResult:
        """Run a single assistant request."""


DEFAULT_ASSISTANT_CASES = (
    AssistantPromptEvalCase(
        name="assistant sample country by code",
        goal="Generate a sample query",
        root_field="country",
        expected_intent="generate_sample",
        expected_text=("country(code:", "code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant sample countries list",
        goal="Generate a sample query",
        root_field="countries",
        expected_intent="generate_sample",
        expected_text=("countries", "code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant fixes selected field typo",
        goal="Troubleshoot this GraphQL operation",
        root_field="country",
        graphql_call="""
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
    native
    emoji1
    capital
    currency
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
""",
        expected_intent="troubleshoot",
        expected_text=("country(code:", "code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant rejects unsupported goal",
        goal="sdfdsfdf",
        root_field="country",
        graphql_call="query CountryQuery($code: ID!) { country(code: $code) { code name native emoji1 capital currency continent { code name } languages { code name } } }",
        expected_intent="unsupported",
        expected_error_text="Assistant goal must ask to generate a sample GraphQL operation or troubleshoot a GraphQL operation.",
    ),
)

DEFAULT_SAMPLE_CASES = tuple(
    SamplePromptEvalCase(
        name=case.name,
        root_field=case.root_field,
        expected_text=case.expected_text,
    )
    for case in DEFAULT_ASSISTANT_CASES
    if case.expected_intent == "generate_sample"
)

DEFAULT_TROUBLESHOOTING_CASES = tuple(
    TroubleshootingPromptEvalCase(
        name=case.name,
        root_field=case.root_field,
        graphql_call=case.graphql_call or "",
        expected_suggestion_text=case.expected_text,
    )
    for case in DEFAULT_ASSISTANT_CASES
    if case.expected_intent == "troubleshoot"
)


def run_assistant_prompt_eval_cases(
    assistant: AssistantRunner,
    schema_file: object,
    cases: Iterable[AssistantPromptEvalCase] = DEFAULT_ASSISTANT_CASES,
) -> list[PromptEvalResult]:
    """Run end-to-end prompt eval cases through the assistant workflow.

    This is the most relevant eval for the current architecture because it
    covers both the Agno-backed intent planner and the selected downstream
    tool. The scoring remains intentionally simple:

    - verify the assistant chooses the expected workflow,
    - score the returned generation or troubleshooting result with existing
      GraphQL guardrails,
    - and verify unsupported goals are rejected explicitly.
    """
    results = []

    for case in cases:
        request = GraphQLAssistantGoal(
            goal=case.goal,
            root_field=case.root_field,
            graphql_call=case.graphql_call,
        )
        try:
            result = assistant.run(request)
            if not isinstance(result, GraphQLAssistantResult):
                raise TypeError(
                    "Assistant runner must return GraphQLAssistantResult, "
                    f"got {type(result).__name__}."
                )

            checks = list(_score_assistant_intent(case, result.intent))
            checks.extend(_score_assistant_output(case, result.output, schema_file))
            results.append(PromptEvalResult("assistant", case.name, _all_checks_passed(tuple(checks)), tuple(checks)))
        except AgentPlanningError as exc:
            if case.expected_intent == "unsupported":
                checks = (
                    _format_check(
                        "assistant rejects unsupported goal",
                        case.expected_error_text in str(exc) if case.expected_error_text else True,
                        None if not case.expected_error_text or case.expected_error_text in str(exc) else [str(exc)],
                    ),
                )
                results.append(PromptEvalResult("assistant", case.name, _all_checks_passed(checks), checks))
            else:
                results.append(PromptEvalResult("assistant", case.name, False, (), str(exc)))
        except TypeError:
            raise
        except Exception as exc:
            results.append(PromptEvalResult("assistant", case.name, False, (), str(exc)))

    return results


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
    parser = argparse.ArgumentParser(description="Run assistant prompt evaluation cases.")
    parser.add_argument(
        "--intent",
        "--workflow",
        dest="intent",
        choices=("all", "generate_sample", "troubleshoot", "unsupported"),
        default="all",
        help="Assistant intent scenarios to evaluate.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the Chroma schema index before evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run prompt evaluation cases against the public assistant workflow."""
    args = parse_args()
    sample_tool = SampleTool(rebuild_index=args.rebuild)
    sample_tool.llm_pre_warmer.pre_warm()
    troubleshooting_tool = TroubleshootingTool(
        settings=sample_tool.settings,
        llm_client=sample_tool.llm_client,
        llm_pre_warmer=sample_tool.llm_pre_warmer,
        schema_context_provider=sample_tool.schema_context_provider,
    )
    assistant = GraphQLAssistantAgent(
        llm_client=sample_tool.llm_client,
        sample_tool=sample_tool,
        troubleshooting_tool=troubleshooting_tool,
    )
    results = run_assistant_prompt_eval_cases(
        assistant,
        sample_tool.settings.schema_file,
        _assistant_cases_for_intent(args.intent),
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
    checks.append(_check_requested_root_field(sample.operation, case.root_field))

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
    original_validation_errors = validate_operation_against_schema(case.graphql_call, schema_file)
    checks = [
        _format_check(
            "submitted operation is invalid before troubleshooting",
            bool(original_validation_errors),
            original_validation_errors or ["submitted operation unexpectedly validated"],
        ),
        _format_check("original call reports validation issues", bool(result.issues)),
        _format_check("detail explains the correction", bool(result.detail)),
        _format_check("suggestion is returned", bool(result.suggestion.strip())),
        _format_check(
            "result status matches troubleshooting outcome",
            result.status == ("invalid" if result.issues else "valid"),
            [f"status was `{result.status}` for issues={bool(result.issues)}"]
            if result.status != ("invalid" if result.issues else "valid")
            else None,
        ),
    ]

    if result.suggestion:
        validation_errors = validate_operation_against_schema(result.suggestion, schema_file)
        checks.append(_format_check("suggestion validates against schema", not validation_errors, validation_errors))
        checks.append(_check_requested_root_field(result.suggestion, case.root_field))
    else:
        checks.append(_format_check("suggestion validates against schema", False, ["missing suggestion"]))
        checks.append(_format_check(f"suggestion targets requested root field `{case.root_field}`", False, ["missing suggestion"]))

    for expected_text in case.expected_suggestion_text:
        checks.append(
            _format_check(
                f"suggestion contains `{expected_text}`",
                expected_text in result.suggestion,
            )
        )

    return tuple(checks)


def _assistant_cases_for_intent(intent: str) -> tuple[AssistantPromptEvalCase, ...]:
    if intent == "generate_sample":
        return tuple(case for case in DEFAULT_ASSISTANT_CASES if case.expected_intent == "generate_sample")
    if intent == "troubleshoot":
        return tuple(case for case in DEFAULT_ASSISTANT_CASES if case.expected_intent == "troubleshoot")
    if intent == "unsupported":
        return tuple(case for case in DEFAULT_ASSISTANT_CASES if case.expected_intent == "unsupported")

    return DEFAULT_ASSISTANT_CASES


def _score_assistant_intent(case: AssistantPromptEvalCase, actual_intent: str) -> tuple[str, ...]:
    return (
        _format_check(
            f"assistant selects `{case.expected_intent}` intent",
            actual_intent == case.expected_intent,
            [f"actual intent was `{actual_intent}`"] if actual_intent != case.expected_intent else None,
        ),
    )


def _score_assistant_output(
    case: AssistantPromptEvalCase,
    output: GeneratedGraphQLSample | TroubleshootingResult | None,
    schema_file: object,
) -> tuple[str, ...]:
    if case.expected_intent == "generate_sample":
        if not isinstance(output, GeneratedGraphQLSample):
            return (
                _format_check(
                    "assistant returns a sample result",
                    False,
                    [f"actual output type was `{type(output).__name__}`"],
                ),
            )

        sample_case = SamplePromptEvalCase(
            name=case.name,
            root_field=case.root_field,
            expected_text=case.expected_text,
        )
        return _score_sample(sample_case, output, schema_file)

    if case.expected_intent == "troubleshoot":
        if not isinstance(output, TroubleshootingResult):
            return (
                _format_check(
                    "assistant returns a troubleshooting result",
                    False,
                    [f"actual output type was `{type(output).__name__}`"],
                ),
            )

        troubleshooting_case = TroubleshootingPromptEvalCase(
            name=case.name,
            root_field=case.root_field,
            graphql_call=case.graphql_call or "",
            expected_suggestion_text=case.expected_text,
        )
        return _score_troubleshooting(troubleshooting_case, output, schema_file)

    return (_format_check("assistant should reject unsupported goal", False),)


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


def _check_requested_root_field(operation: str, root_field: str) -> str:
    operation_root_fields, parse_errors = _extract_root_fields(operation)
    if parse_errors:
        return _format_check(
            f"operation targets requested root field `{root_field}`",
            False,
            parse_errors,
        )

    return _format_check(
        f"operation targets requested root field `{root_field}`",
        root_field in operation_root_fields,
        [f"top-level root fields were: {', '.join(operation_root_fields) or '<none>'}"]
        if root_field not in operation_root_fields
        else None,
    )


def _extract_root_fields(operation: str) -> tuple[tuple[str, ...], list[str] | None]:
    try:
        from graphql import OperationDefinitionNode, parse
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

    try:
        document = parse(operation)
    except Exception as exc:
        return (), [str(exc).split("\n\n", maxsplit=1)[0]]

    root_fields: list[str] = []
    for definition in document.definitions:
        if not isinstance(definition, OperationDefinitionNode):
            continue
        for selection in definition.selection_set.selections:
            field_name = getattr(getattr(selection, "name", None), "value", None)
            if field_name:
                root_fields.append(field_name)

    return tuple(root_fields), None


if __name__ == "__main__":
    main()
