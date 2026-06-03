from __future__ import annotations

import hashlib
import re
from pathlib import Path

from graphql_ai.domain import SchemaChunk


DEFINITION_START = re.compile(
    r"(?m)^\s*(?:extend\s+)?"
    r"(schema|type|input|enum|interface|union|scalar|directive)\b"
    r"(?:\s+([_A-Za-z][_0-9A-Za-z]*))?"
)
FIELD_START = re.compile(r"^\s*([_A-Za-z][_0-9A-Za-z]*)\s*(?:\(|:)")


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

        if kind == "type" and name in {"Query", "Mutation"}:
            root_chunks = chunk_root_operation(definition_text, source, name)
            if root_chunks:
                chunks.extend(root_chunks)
                continue

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


def chunk_root_operation(definition_text: str, source: str, operation_name: str) -> list[SchemaChunk]:
    """Split Query or Mutation definitions into field-level chunks."""
    body = extract_braced_body(definition_text)
    field_chunks = []

    for field_text in split_root_fields(body):
        field_name = get_field_name(field_text)
        if field_name is None:
            continue

        kind = operation_name.lower()
        text = f"type {operation_name} {{\n{indent_field_text(field_text)}\n}}"
        field_chunks.append(
            SchemaChunk(
                id=make_chunk_id(source, kind, field_name, text),
                source=source,
                kind=kind,
                name=field_name,
                text=text,
            )
        )

    return field_chunks


def extract_braced_body(definition_text: str) -> str:
    """Extract the text inside the outermost SDL definition braces."""
    start = definition_text.find("{")
    end = definition_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    return definition_text[start + 1 : end]


def split_root_fields(body: str) -> list[str]:
    """Split a root operation body into individual field definitions."""
    fields: list[str] = []
    current: list[str] = []
    paren_depth = 0

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        starts_new_field = bool(FIELD_START.match(line)) and paren_depth == 0
        if starts_new_field and current:
            current_text = "\n".join(current).strip()
            if get_field_name(current_text) is not None:
                fields.append(current_text)
                current = []

        current.append(line)
        paren_depth += line.count("(") - line.count(")")

    if current:
        current_text = "\n".join(current).strip()
        if get_field_name(current_text) is not None:
            fields.append(current_text)

    return fields


def get_field_name(field_text: str) -> str | None:
    """Return the GraphQL field name from a field definition."""
    for line in field_text.splitlines():
        match = FIELD_START.match(line)
        if match is not None:
            return match.group(1)

    return None


def indent_field_text(field_text: str) -> str:
    """Normalize a field definition to two-space indentation."""
    return "\n".join(f"  {line.lstrip()}" for line in field_text.splitlines())


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
