from __future__ import annotations

import argparse

from graphql_assistant.agents import GraphQLAssistantAgent, GraphQLAssistantGoal
from graphql_assistant.agents.tools import SampleTool, TroubleshootingTool
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_assistant.core.config import get_settings
from graphql_assistant.llm.factory import build_llm_client
from graphql_assistant.llm.pre_warm import LLMPreWarmer
from graphql_assistant.rag.vector_store import SchemaVectorStore


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the assistant CLI."""
    parser = argparse.ArgumentParser(description="Run the GraphQL assistant from the command line.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Generate a sample query",
        help="Natural-language assistant goal, for example: Generate a sample query.",
    )
    parser.add_argument(
        "root_field",
        nargs="?",
        default="country",
        help="GraphQL Query or Mutation field name the assistant should focus on.",
    )
    parser.add_argument(
        "--graphql-call",
        help="GraphQL operation to troubleshoot when the goal is a troubleshooting request.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the Chroma schema index.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the assistant CLI."""
    args = parse_args()
    settings = get_settings()
    schema_context_provider = SchemaVectorStore(settings=settings, rebuild=args.rebuild)
    llm_client = build_llm_client(settings)
    llm_pre_warmer = LLMPreWarmer(settings, llm_client)
    llm_pre_warmer.pre_warm()

    sample_tool = SampleTool(
        settings=settings,
        llm_client=llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=schema_context_provider,
    )
    troubleshooting_tool = TroubleshootingTool(
        settings=settings,
        llm_client=llm_client,
        llm_pre_warmer=llm_pre_warmer,
        schema_context_provider=schema_context_provider,
    )
    assistant = GraphQLAssistantAgent(
        llm_client=llm_client,
        sample_tool=sample_tool,
        troubleshooting_tool=troubleshooting_tool,
    )
    result = assistant.run(
        GraphQLAssistantGoal(
            goal=args.goal,
            root_field=args.root_field,
            graphql_call=args.graphql_call,
        )
    )

    print(f"\nAssistant intent: {result.intent}\n")
    _print_output(result.output)


def _print_output(output: GeneratedGraphQLSample | TroubleshootingResult) -> None:
    if isinstance(output, GeneratedGraphQLSample):
        print("Generated result:\n")
        print(output.raw_response)
        return

    print("Troubleshooting result:\n")
    print(f"Status: {output.status}")
    if output.issues:
        print("Issues:")
        for issue in output.issues:
            print(f"- {issue}")
    if output.detail:
        print("Detail:")
        for line in output.detail:
            print(f"- {line}")
    if output.suggestion:
        print("Suggestion:\n")
        print(output.suggestion)


if __name__ == "__main__":
    main()
