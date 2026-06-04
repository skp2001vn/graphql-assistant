from __future__ import annotations

import hashlib
import re
from pathlib import Path

from graphql_ai.domain import SchemaChunk


SCHEMA_CHUNK_VERSION = "2"
DEFINITION_START = re.compile(
    r"(?m)^\s*(?:extend\s+)?"
    r"(schema|type|input|enum|interface|union|scalar|directive)\b"
    r"(?:\s+([_A-Za-z][_0-9A-Za-z]*))?"
)


def read_schema_file(schema_file: Path) -> tuple[str, str]:
    """Read a local GraphQL SDL file and return its source path and text."""
    if not schema_file.exists():
        raise FileNotFoundError(f"Local GraphQL schema file not found: {schema_file}")

    schema_text = schema_file.read_text(encoding="utf-8").strip()
    if not schema_text:
        raise ValueError(f"Local schema was empty: {schema_file}")

    return str(schema_file), schema_text


def load_schema_chunks(schema_file: Path) -> list[SchemaChunk]:
    """Load and chunk a local GraphQL schema file."""
    source, schema_text = read_schema_file(schema_file)
    chunks = chunk_graphql_schema(schema_text, source)

    if not chunks:
        raise ValueError(f"No schema chunks were created from {source}")

    return chunks


def chunk_graphql_schema(schema_text: str, source: str) -> list[SchemaChunk]:
    """Split GraphQL SDL into retrievable schema chunks."""
    matches = list(DEFINITION_START.finditer(schema_text))

    if not matches:
        text = schema_text.strip()
        return [
            SchemaChunk(
                id=make_chunk_id(source, "file", source, text),
                source=source,
                kind="file",
                name=source,
                text=text,
            )
        ]

    chunks: list[SchemaChunk] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(schema_text)
        definition_text = schema_text[start:end].strip()
        kind = match.group(1)
        name = match.group(2) or kind

        chunks.append(
            SchemaChunk(
                id=make_chunk_id(source, kind, name, definition_text),
                source=source,
                kind=kind,
                name=name,
                text=definition_text,
            )
        )

    return dedupe_chunks(chunks)


def make_chunk_id(source: str, kind: str, name: str, text: str) -> str:
    """Create a stable short ID for a schema chunk."""
    digest = hashlib.sha1(f"{source}:{kind}:{name}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def dedupe_chunks(chunks: list[SchemaChunk]) -> list[SchemaChunk]:
    """Remove duplicate chunks while preserving order."""
    unique_chunks = []
    seen_ids = set()

    for chunk in chunks:
        if chunk.id in seen_ids:
            continue

        seen_ids.add(chunk.id)
        unique_chunks.append(chunk)

    return unique_chunks
