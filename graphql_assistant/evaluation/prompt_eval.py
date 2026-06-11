from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from graphql_assistant.agents import (
    AgnoAssistantPlanner,
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
from graphql_assistant.llm.pre_warm import LLMPreWarmer


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
    expected_operation_name: str | None = None
    expected_variable_names: tuple[str, ...] = ()
    expected_root_field_arguments: tuple[str, ...] = ()
    expected_root_field_selections: tuple[str, ...] = ()
    graphql_call: str | None = None
    expected_status: Literal["valid", "invalid"] | None = None
    expected_error_text: str = ""


@dataclass(frozen=True)
class SamplePromptEvalCase:
    """Prompt evaluation case for sample GraphQL generation."""

    name: str
    root_field: str
    expected_operation_name: str | None = None
    expected_variable_names: tuple[str, ...] = ()
    expected_root_field_arguments: tuple[str, ...] = ()
    expected_root_field_selections: tuple[str, ...] = ()


@dataclass(frozen=True)
class TroubleshootingPromptEvalCase:
    """Prompt evaluation case for GraphQL troubleshooting."""

    name: str
    root_field: str
    graphql_call: str
    expected_operation_name: str | None = None
    expected_variable_names: tuple[str, ...] = ()
    expected_root_field_arguments: tuple[str, ...] = ()
    expected_root_field_selections: tuple[str, ...] = ()
    expected_status: Literal["valid", "invalid"] = "invalid"


@dataclass(frozen=True)
class PromptEvalResult:
    """Prompt evaluation result with simple pass/fail checks."""

    workflow: str
    name: str
    passed: bool
    checks: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class OperationShape:
    """Parsed GraphQL operation properties used for prompt eval assertions."""

    operation_name: str | None
    root_field_name: str | None
    variable_names: tuple[str, ...]
    root_field_arguments: tuple[str, ...]
    root_field_selections: tuple[str, ...]


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
        expected_operation_name="CountryQuery",
        expected_variable_names=("code",),
        expected_root_field_arguments=("code",),
        expected_root_field_selections=("code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant sample countries list",
        goal="Generate a sample query",
        root_field="countries",
        expected_intent="generate_sample",
        expected_operation_name="CountriesQuery",
        expected_root_field_selections=("code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant sample continent by code",
        goal="Generate a sample query",
        root_field="continent",
        expected_intent="generate_sample",
        expected_operation_name="ContinentQuery",
        expected_variable_names=("code",),
        expected_root_field_arguments=("code",),
        expected_root_field_selections=("code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant sample continents list",
        goal="Generate a sample query",
        root_field="continents",
        expected_intent="generate_sample",
        expected_operation_name="ContinentsQuery",
        expected_root_field_selections=("code", "name"),
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
        expected_status="invalid",
        expected_operation_name="CountryQuery",
        expected_variable_names=("code",),
        expected_root_field_arguments=("code",),
        expected_root_field_selections=("code", "name"),
    ),
    AssistantPromptEvalCase(
        name="assistant recognizes valid country query",
        goal="Troubleshoot this GraphQL operation",
        root_field="country",
        graphql_call="""
query CountryQuery($code: ID!) {
  country(code: $code) {
    code
    name
  }
}
""",
        expected_intent="troubleshoot",
        expected_status="valid",
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
        expected_operation_name=case.expected_operation_name,
        expected_variable_names=case.expected_variable_names,
        expected_root_field_arguments=case.expected_root_field_arguments,
        expected_root_field_selections=case.expected_root_field_selections,
    )
    for case in DEFAULT_ASSISTANT_CASES
    if case.expected_intent == "generate_sample"
)

DEFAULT_TROUBLESHOOTING_CASES = tuple(
    TroubleshootingPromptEvalCase(
        name=case.name,
        root_field=case.root_field,
        graphql_call=case.graphql_call or "",
        expected_operation_name=case.expected_operation_name,
        expected_variable_names=case.expected_variable_names,
        expected_root_field_arguments=case.expected_root_field_arguments,
        expected_root_field_selections=case.expected_root_field_selections,
        expected_status=case.expected_status or "invalid",
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
    llm_pre_warmer = LLMPreWarmer(sample_tool.settings, sample_tool.llm_client)
    llm_pre_warmer.pre_warm()
    troubleshooting_tool = TroubleshootingTool(
        settings=sample_tool.settings,
        llm_client=sample_tool.llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=sample_tool.schema_context_provider,
    )
    assistant = GraphQLAssistantAgent(
        sample_tool=sample_tool,
        troubleshooting_tool=troubleshooting_tool,
        planner=AgnoAssistantPlanner(sample_tool.llm_client),
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
    checks.extend(_score_operation_shape(sample.operation, case))

    return tuple(checks)


def _score_troubleshooting(
    case: TroubleshootingPromptEvalCase,
    result: TroubleshootingResult,
    schema_file: object,
) -> tuple[str, ...]:
    original_validation_errors = validate_operation_against_schema(case.graphql_call, schema_file)
    expected_status = case.expected_status
    is_valid_passthrough = expected_status == "valid"
    checks = [
        _format_check(
            "submitted operation matches expected troubleshooting precondition",
            (not original_validation_errors) if is_valid_passthrough else bool(original_validation_errors),
            ["submitted operation was expected to validate"]
            if is_valid_passthrough and original_validation_errors
            else ["submitted operation unexpectedly validated"]
            if not is_valid_passthrough and not original_validation_errors
            else original_validation_errors,
        ),
        _format_check(
            f"result status is `{expected_status}`",
            result.status == expected_status,
            [f"status was `{result.status}`"] if result.status != expected_status else None,
        ),
        _format_check(
            "result status matches troubleshooting outcome",
            result.status == ("invalid" if result.issues else "valid"),
            [f"status was `{result.status}` for issues={bool(result.issues)}"]
            if result.status != ("invalid" if result.issues else "valid")
            else None,
        ),
    ]

    if is_valid_passthrough:
        checks.extend(
            [
                _format_check("valid result reports no issues", not result.issues, result.issues or None),
                _format_check("valid result keeps detail empty", not result.detail),
                _format_check("valid result keeps suggestion empty", not result.suggestion.strip()),
            ]
        )
        return tuple(checks)

    checks.extend(
        [
            _format_check("original call reports validation issues", bool(result.issues)),
            _format_check("detail explains the correction", bool(result.detail)),
            _format_check("suggestion is returned", bool(result.suggestion.strip())),
        ]
    )

    if result.suggestion:
        validation_errors = validate_operation_against_schema(result.suggestion, schema_file)
        checks.append(_format_check("suggestion validates against schema", not validation_errors, validation_errors))
        checks.extend(_score_operation_shape(result.suggestion, case))
    else:
        checks.append(_format_check("suggestion validates against schema", False, ["missing suggestion"]))
        checks.append(_format_check(f"suggestion targets requested root field `{case.root_field}`", False, ["missing suggestion"]))

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
            expected_operation_name=case.expected_operation_name,
            expected_variable_names=case.expected_variable_names,
            expected_root_field_arguments=case.expected_root_field_arguments,
            expected_root_field_selections=case.expected_root_field_selections,
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
            expected_operation_name=case.expected_operation_name,
            expected_variable_names=case.expected_variable_names,
            expected_root_field_arguments=case.expected_root_field_arguments,
            expected_root_field_selections=case.expected_root_field_selections,
            expected_status=case.expected_status or "invalid",
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


def _score_operation_shape(
    operation: str,
    case: SamplePromptEvalCase | TroubleshootingPromptEvalCase,
) -> tuple[str, ...]:
    shape, parse_errors = _extract_operation_shape(operation, case.root_field)
    if parse_errors:
        return (
            _format_check(
                f"operation targets requested root field `{case.root_field}`",
                False,
                parse_errors,
            ),
        )

    checks = [_check_requested_root_field_shape(case, shape)]

    if case.expected_operation_name is not None:
        checks.append(
            _format_check(
                f"operation name is `{case.expected_operation_name}`",
                shape.operation_name == case.expected_operation_name,
                [f"operation name was `{shape.operation_name or '<anonymous>'}`"]
                if shape.operation_name != case.expected_operation_name
                else None,
            )
        )

    checks.append(
        _format_check(
            f"declared variables are {case.expected_variable_names or 'empty'}",
            shape.variable_names == case.expected_variable_names,
            [f"variables were: {shape.variable_names or '<empty>'}"]
            if shape.variable_names != case.expected_variable_names
            else None,
        )
    )
    checks.append(
        _format_check(
            f"root field arguments are {case.expected_root_field_arguments or 'empty'}",
            shape.root_field_arguments == case.expected_root_field_arguments,
            [f"arguments were: {shape.root_field_arguments or '<empty>'}"]
            if shape.root_field_arguments != case.expected_root_field_arguments
            else None,
        )
    )

    for selection_name in case.expected_root_field_selections:
        checks.append(
            _format_check(
                f"root field selects `{selection_name}`",
                selection_name in shape.root_field_selections,
            )
        )

    return tuple(checks)


def _check_requested_root_field_shape(
    case: SamplePromptEvalCase | TroubleshootingPromptEvalCase,
    shape: OperationShape,
) -> str:
    return _format_check(
        f"operation targets requested root field `{case.root_field}`",
        shape.root_field_name == case.root_field,
        [f"top-level root field was `{shape.root_field_name or '<none>'}`"]
        if shape.root_field_name != case.root_field
        else None,
    )


def _extract_operation_shape(operation: str, root_field: str) -> tuple[OperationShape, list[str] | None]:
    try:
        from graphql import FieldNode, OperationDefinitionNode, parse
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install graphql-core with `pip install -r requirements.txt`.") from exc

    try:
        document = parse(operation)
    except Exception as exc:
        return OperationShape(None, None, (), (), ()), [str(exc).split("\n\n", maxsplit=1)[0]]

    operation_name: str | None = None
    variable_names: tuple[str, ...] = ()
    root_field_name: str | None = None
    root_field_arguments: tuple[str, ...] = ()
    root_field_selections: tuple[str, ...] = ()

    for definition in document.definitions:
        if not isinstance(definition, OperationDefinitionNode):
            continue
        operation_name = getattr(getattr(definition, "name", None), "value", None)
        variable_names = tuple(variable.variable.name.value for variable in definition.variable_definitions or ())

        for selection in definition.selection_set.selections:
            if not isinstance(selection, FieldNode):
                continue
            field_name = selection.name.value
            if field_name != root_field:
                if root_field_name is None:
                    root_field_name = field_name
                continue

            root_field_name = field_name
            root_field_arguments = tuple(argument.name.value for argument in selection.arguments or ())
            root_field_selections = tuple(
                nested_selection.name.value
                for nested_selection in selection.selection_set.selections
                if isinstance(nested_selection, FieldNode)
            ) if selection.selection_set is not None else ()
            return OperationShape(
                operation_name=operation_name,
                root_field_name=root_field_name,
                variable_names=variable_names,
                root_field_arguments=root_field_arguments,
                root_field_selections=root_field_selections,
            ), None

        break

    return OperationShape(
        operation_name=operation_name,
        root_field_name=root_field_name,
        variable_names=variable_names,
        root_field_arguments=root_field_arguments,
        root_field_selections=root_field_selections,
    ), None


if __name__ == "__main__":
    main()
