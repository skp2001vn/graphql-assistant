from __future__ import annotations

import argparse

from graphql_ai.services.sample_query_service import SampleQueryService


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample-query generation."""
    parser = argparse.ArgumentParser(description="Generate sample GraphQL calls with the local AI pipeline.")
    parser.add_argument(
        "request",
        nargs="?",
        default="Generate a sample query for a country by code",
        help="Natural-language request for the GraphQL sample call.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the Chroma schema index.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the sample-query CLI."""
    args = parse_args()
    sample_service = SampleQueryService(rebuild_index=args.rebuild)
    sample = sample_service.generate(args.request)

    print("\nGenerated result:\n")
    print(sample.raw_response)


if __name__ == "__main__":
    main()
