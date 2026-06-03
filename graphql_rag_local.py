from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA_FILE = Path("resources/schema.graphql")
SCHEMA_FILE = Path(os.getenv("GRAPHQL_SCHEMA_FILE", str(DEFAULT_SCHEMA_FILE)))
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "graphql_schema")
CACHE_METADATA_FILE = "index_metadata.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "resources/models/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1200"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() in {"1", "true", "yes"}

GRAPHQL_SYSTEM_PROMPT = """You are a GraphQL expert.

Generate a valid GraphQL query or mutation from the provided schema context.

Rules:
- Use only fields and arguments that exist in the schema.
- Include all fields for the requested response type.
- Expand nested objects and lists using their schema fields.
- Use variables for argument values.
- Return only two fenced code blocks with no labels or headings: first GraphQL operation, second Variables JSON.
"""

GRAPHQL_USER_PROMPT_TEMPLATE = """Schema:

{schema_context}

Request:

{user_request}

Return:
1. GraphQL operation in a code block
2. Variables JSON in a code block
"""

DEFINITION_START = re.compile(
    r"(?m)^\s*(?:extend\s+)?"
    r"(schema|type|input|enum|interface|union|scalar|directive)\b"
    r"(?:\s+([_A-Za-z][_0-9A-Za-z]*))?"
)
FIELD_START = re.compile(r"^\s*([_A-Za-z][_0-9A-Za-z]*)\s*(?:\(|:)")
VARIABLE_DEFINITION = re.compile(r"\$([_A-Za-z][_0-9A-Za-z]*)\s*:\s*([!\[\]_0-9A-Za-z]+)")


@dataclass(frozen=True)
class SchemaChunk:
    id: str
    source: str
    kind: str
    name: str
    text: str


@dataclass(frozen=True)
class GeneratedGraphQLSample:
    operation: str
    variables: dict[str, Any]
    raw_response: str


_embedding_model: tuple[str, Any] | None = None


def get_embedding_model(model_name_or_path: str = EMBEDDING_MODEL, allow_downloads: bool = False) -> Any:
    global _embedding_model

    if _embedding_model is None or _embedding_model[0] != model_name_or_path:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install sentence-transformers with "
                "`pip install -r requirements.txt`."
            ) from exc

        model_path = Path(model_name_or_path)
        if not allow_downloads and _looks_like_local_path(model_name_or_path) and not model_path.exists():
            raise RuntimeError(
                f"Local embedding model not found: {model_path}\n"
                "Download it once while online, then run again locally:\n"
                "  python -c \"from sentence_transformers import SentenceTransformer; "
                "SentenceTransformer('all-MiniLM-L6-v2').save_pretrained("
                "'resources/models/all-MiniLM-L6-v2')\""
            )

        _embedding_model = (
            model_name_or_path,
            SentenceTransformer(model_name_or_path, local_files_only=not allow_downloads),
        )

    return _embedding_model[1]


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or "/" in value or "\\" in value


def read_schema_file(schema_file: Path = SCHEMA_FILE) -> tuple[str, str]:
    if not schema_file.exists():
        raise FileNotFoundError(f"Local GraphQL schema file not found: {schema_file}")

    schema_text = schema_file.read_text(encoding="utf-8").strip()
    if not schema_text:
        raise ValueError(f"Local schema was empty: {schema_file}")

    return str(schema_file), schema_text


def make_chunk_id(source: str, kind: str, name: str, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{kind}:{name}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def make_schema_fingerprint(schema_file: Path, embedding_model: str) -> dict[str, Any]:
    schema_text = schema_file.read_text(encoding="utf-8")
    return {
        "schema_file": str(schema_file),
        "schema_sha1": hashlib.sha1(schema_text.encode("utf-8")).hexdigest(),
        "embedding_model": embedding_model,
        "collection_name": COLLECTION_NAME,
    }


def cache_metadata_path() -> Path:
    return Path(CHROMA_PATH) / CACHE_METADATA_FILE


def read_cache_metadata() -> dict[str, Any] | None:
    path = cache_metadata_path()
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_cache_metadata(metadata: dict[str, Any]) -> None:
    path = cache_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def has_valid_cached_index(schema_file: Path, embedding_model: str) -> bool:
    return read_cache_metadata() == make_schema_fingerprint(schema_file, embedding_model)


def extract_braced_body(definition_text: str) -> str:
    start = definition_text.find("{")
    end = definition_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    return definition_text[start + 1 : end]


def split_root_fields(body: str) -> list[str]:
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
    for line in field_text.splitlines():
        match = FIELD_START.match(line)
        if match is not None:
            return match.group(1)

    return None


def indent_field_text(field_text: str) -> str:
    return "\n".join(f"  {line.lstrip()}" for line in field_text.splitlines())


def chunk_root_operation(definition_text: str, source: str, operation_name: str) -> list[SchemaChunk]:
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


def chunk_graphql_schema(schema_text: str, source: str) -> list[SchemaChunk]:
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


def dedupe_chunks(chunks: list[SchemaChunk]) -> list[SchemaChunk]:
    unique_chunks = []
    seen_ids = set()

    for chunk in chunks:
        if chunk.id in seen_ids:
            continue

        seen_ids.add(chunk.id)
        unique_chunks.append(chunk)

    return unique_chunks


def load_schema_chunks(schema_file: Path = SCHEMA_FILE) -> list[SchemaChunk]:
    source, schema_text = read_schema_file(schema_file)
    chunks = chunk_graphql_schema(schema_text, source)

    if not chunks:
        raise ValueError(f"No schema chunks were created from {source}")

    return chunks


def embed_texts(
    texts: list[str],
    model_name_or_path: str = EMBEDDING_MODEL,
    allow_downloads: bool = False,
) -> list[list[float]]:
    embeddings = get_embedding_model(model_name_or_path, allow_downloads).encode(
        texts,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def build_vector_store(
    schema_file: Path = SCHEMA_FILE,
    rebuild: bool = False,
    embedding_model: str = EMBEDDING_MODEL,
    allow_downloads: bool = False,
):
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install chromadb with `pip install -r requirements.txt`."
        ) from exc

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    if not rebuild and has_valid_cached_index(schema_file, embedding_model):
        collection = client.get_or_create_collection(COLLECTION_NAME)
        if collection.count() > 0:
            print(f"Using cached GraphQL schema index from {CHROMA_PATH}.")
            return collection

        print("Cached schema index metadata exists, but the Chroma collection is empty; rebuilding.")
    elif not rebuild:
        print("Schema index cache is missing or stale; rebuilding.")

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    chunks = load_schema_chunks(schema_file)
    documents = [chunk.text for chunk in chunks]

    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=documents,
        embeddings=embed_texts(documents, embedding_model, allow_downloads),
        metadatas=[
            {
                "source": chunk.source,
                "kind": chunk.kind,
                "name": chunk.name,
            }
            for chunk in chunks
        ],
    )

    write_cache_metadata(make_schema_fingerprint(schema_file, embedding_model))
    print(f"Indexed {len(chunks)} GraphQL schema chunks from {schema_file}.")
    return collection


def retrieve_schema_context(
    collection,
    user_request: str,
    embedding_model: str = EMBEDDING_MODEL,
    allow_downloads: bool = False,
) -> str:
    results = collection.query(
        query_embeddings=embed_texts([user_request], embedding_model, allow_downloads),
        n_results=collection.count(),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []
    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "unknown")
        kind = metadata.get("kind", "definition")
        name = metadata.get("name", "unknown")
        context_parts.append(f"# Source: {source} ({kind} {name})\n{document}")

    return "\n\n".join(context_parts)


def call_ollama(prompt: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install requests with `pip install -r requirements.txt`.") from exc

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": OLLAMA_THINK,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model or endpoint not found.\n"
                f"Configured model: {OLLAMA_MODEL}\n"
                f"Pull the model first with:\n"
                f"  ollama pull {OLLAMA_MODEL}\n"
                f"Ollama response: {response.text}"
            ) from exc

        raise RuntimeError(f"Ollama request failed: {response.text}") from exc

    return str(response.json().get("response", "")).strip()


def generate_graphql_sample(
    collection,
    user_request: str,
    embedding_model: str = EMBEDDING_MODEL,
    allow_downloads: bool = False,
) -> str:
    schema_context = retrieve_schema_context(
        collection,
        user_request,
        embedding_model=embedding_model,
        allow_downloads=allow_downloads,
    )
    user_prompt = GRAPHQL_USER_PROMPT_TEMPLATE.format(
        schema_context=schema_context,
        user_request=user_request,
    )
    return call_ollama(f"{GRAPHQL_SYSTEM_PROMPT}\n\n{user_prompt}")


def parse_generated_sample(raw_response: str) -> GeneratedGraphQLSample:
    code_blocks = re.findall(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", raw_response, flags=re.DOTALL)
    operation = code_blocks[0].strip() if code_blocks else raw_response.strip()
    variables: dict[str, Any] = {}

    if len(code_blocks) > 1:
        variables_text = code_blocks[1].strip()
        if variables_text:
            try:
                parsed_variables = json.loads(variables_text)
            except json.JSONDecodeError:
                parsed_variables = {"_raw": variables_text}

            if isinstance(parsed_variables, dict):
                variables = parsed_variables
            else:
                variables = {"value": parsed_variables}

    if not variables:
        variables = infer_variables_from_operation(operation)

    return GeneratedGraphQLSample(
        operation=operation,
        variables=variables,
        raw_response=raw_response,
    )


def infer_variables_from_operation(operation: str) -> dict[str, Any]:
    inferred_variables: dict[str, Any] = {}

    for variable_name, type_ref in VARIABLE_DEFINITION.findall(operation):
        inferred_variables[variable_name] = sample_value_for_graphql_type(variable_name, type_ref)

    return inferred_variables


def sample_value_for_graphql_type(variable_name: str, type_ref: str) -> Any:
    base_type = re.sub(r"[\[\]!]", "", type_ref)

    if type_ref.startswith("["):
        return [sample_value_for_graphql_type(variable_name, base_type)]

    if base_type == "Boolean":
        return True
    if base_type == "Float":
        return 1.0
    if base_type == "Int":
        return 1
    if base_type == "ID":
        return "US" if "code" in variable_name.lower() else "example-id"

    return "example"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample GraphQL calls with local RAG.")
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
    args = parse_args()
    collection = build_vector_store(rebuild=args.rebuild)
    result = generate_graphql_sample(collection, args.request)

    print("\nGenerated result:\n")
    print(result)


if __name__ == "__main__":
    main()
