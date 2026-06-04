# GraphQL AI Examples

This is a small app for generating sample GraphQL calls from a local schema.

The current implementation uses RAG. The structure is intentionally open for adding other GraphQL AI capabilities later, such as agents, planning workflows, inference optimization, model routing, and prompt evaluation.

The current RAG flow:

1. Reads `resources/schema.graphql`.
2. Splits the SDL into schema chunks.
3. Embeds those chunks with a local sentence-transformers model.
4. Stores the embeddings in a local Chroma index.
5. Retrieves schema context for your request.
6. Checks the local inference cache for the final prompt.
7. Sends only uncached prompt context to local Ollama and prints the GraphQL operation plus Variables JSON.

Because `resources/schema.graphql` is rarely updated, the Chroma index is cached and reused after the first run.

## One-Time Setup

Run this once while online:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p resources/models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save_pretrained('resources/models/all-MiniLM-L6-v2')"

brew install --cask ollama-app
open -a Ollama
ollama pull qwen2.5-coder:3b
```

On macOS, use `brew install --cask ollama-app`. The `brew install ollama` formula can install without the required `llama-server` runtime binary.

## Run

```bash
source .venv/bin/activate
.venv/bin/python -m graphql_ai.cli "Generate a sample query for a country by code"
```

Example request:

```bash
.venv/bin/python -m graphql_ai.cli "Generate a sample query for all countries"
```

The first run builds the local Chroma index. Later runs reuse it automatically. If you edit `resources/schema.graphql`, the app detects the schema change and rebuilds the index.

To force a rebuild:

```bash
.venv/bin/python -m graphql_ai.cli --rebuild "Generate a sample query for a country by code"
```

## Run The API

Start the FastAPI server:

```bash
source .venv/bin/activate
uvicorn graphql_ai.main:app --host 0.0.0.0 --port 8080
```

Call the sample-query endpoint:

```bash
curl http://localhost:8080/sample/country
```

If port `8080` is already in use, either stop the existing server or use another port:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
kill 94081
```

Replace `94081` with the PID shown by `lsof`.

Or run the API on a different port:

```bash
uvicorn graphql_ai.main:app --host 0.0.0.0 --port 8081
curl http://localhost:8081/sample/country
```

The response is JSON:

```json
{
  "operation": [
    "query CountryQuery($code: ID!) {",
    "  country(code: $code) {",
    "    code",
    "    name",
    "    native",
    "    emoji",
    "    capital",
    "    currency",
    "    continent {",
    "      code",
    "      name",
    "    }",
    "    languages {",
    "      code",
    "      name",
    "    }",
    "  }",
    "}"
  ],
  "variables": {
    "code": "US"
  }
}
```

You can also pass a custom request:

```bash
curl "http://localhost:8080/sample/country?request=Generate%20a%20sample%20query%20for%20all%20countries"
```

The API currently uses RAG. It builds or reuses the Chroma schema index once during application startup, then each endpoint call retrieves schema context and asks Ollama to generate the sample GraphQL call.

## Tests

Run the test suite:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

The tests use fake LLM and schema-context providers, so they do not require Chroma, the embedding model, or a running Ollama server.

## Guardrails

The app applies guardrails after model generation:

- GraphQL-core parses and validates generated operations against `resources/schema.graphql`.
- Invalid output is rejected before the API returns it, including malformed GraphQL, invented fields, missing required arguments, scalar fields with nested selections, and variable type mismatches.
- A separate variable-usage check rejects Variables JSON entries that are not referenced by the GraphQL operation.

## Inference Optimization

The app includes two local caches:

- Chroma schema index cache: avoids re-embedding `resources/schema.graphql` on every run.
- Schema-context cache: avoids re-embedding and querying Chroma for repeated natural-language requests.
- Inference response cache: avoids calling Ollama again when the final prompt and model settings are identical.

The inference cache is useful for local demos because generation is usually the slowest step. It is keyed by the full prompt plus model settings, so changing the request, retrieved schema context, model, `OLLAMA_NUM_PREDICT`, `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, or `OLLAMA_THINK` produces a different cache entry.

Ollama runtime options are also tuned for local responsiveness:

- `OLLAMA_KEEP_ALIVE=10m` keeps the model loaded between requests.
- `OLLAMA_NUM_CTX` optionally controls the context window size.
- `OLLAMA_NUM_PREDICT=600` keeps the maximum output smaller for this simple schema.
- `PROMPT_COMPRESSION_ENABLED=true` keeps schema context and instructions compact before calling Ollama.
- `OLLAMA_PRE_WARM_ENABLED=true` sends a tiny startup request so the model is loaded before the first API call.

Defaults:

```bash
OLLAMA_KEEP_ALIVE=10m
OLLAMA_NUM_CTX=
OLLAMA_NUM_PREDICT=600
OLLAMA_PRE_WARM_ENABLED=true
OLLAMA_PRE_WARM_PROMPT=OK
PROMPT_COMPRESSION_ENABLED=true
SCHEMA_CONTEXT_CACHE_ENABLED=true
SCHEMA_CONTEXT_CACHE_PATH=.cache/schema_context
INFERENCE_CACHE_ENABLED=true
INFERENCE_CACHE_PATH=.cache/inference
```

To disable response caching:

```bash
INFERENCE_CACHE_ENABLED=false uvicorn graphql_ai.main:app --host 0.0.0.0 --port 8080
```

To disable startup model pre-warming:

```bash
OLLAMA_PRE_WARM_ENABLED=false uvicorn graphql_ai.main:app --host 0.0.0.0 --port 8080
```

To clear cached responses:

```bash
rm -rf .cache/inference .cache/schema_context
```

## Project Structure

The application is split into layers instead of keeping everything in one script:

```text
graphql_ai/
  api/
    routes.py          # FastAPI controllers/routes
    schemas.py         # Pydantic request/response models
  core/
    config.py          # Environment-backed application settings
    protocols.py       # Protocol contracts used by services
    responses.py       # Shared response formatting
  domain/
    models.py          # Domain dataclasses shared by services and RAG
  llm/
    base.py            # LLM client protocol
    cache.py           # Prompt/response cache wrapper
    ollama_client.py   # Ollama HTTP client
  rag/
    embeddings.py      # Local embedding model loading
    schema_chunks.py   # GraphQL SDL parsing/chunking
    vector_store.py    # Chroma indexing and retrieval
  services/
    sample_query_service.py # Business service for sample-query generation
  cli.py               # Command-line entry point
  main.py              # FastAPI app factory
tests/                 # Unit and integration tests with fake AI dependencies
  api/
  core/
  domain/
  llm/
  rag/
  services/
  test_cli.py
```

Design notes:

- API routes stay thin and delegate work to the service layer.
- Pydantic schemas define the public HTTP response contract.
- RAG is represented by the `graphql_ai/rag` module, but it is only the current schema-context approach.
- Future approaches such as agents or inference optimization can be added as normal packages when they are implemented.
- The sample-query service depends on a schema-context protocol, so RAG can be swapped or composed with another approach.
- Ollama access is isolated behind a client class and an LLM protocol.
- GraphQL-core validation is used as an output guardrail before generated samples are returned.
- Application settings are centralized in `graphql_ai/core/config.py`.
- The Chroma collection is initialized once during FastAPI startup instead of being rebuilt per request.
- Local generation is serialized with a lock because the embedding model and Ollama call are expensive shared resources.

## Defaults

```bash
GRAPHQL_SCHEMA_FILE=resources/schema.graphql
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=graphql_schema
EMBEDDING_MODEL=resources/models/all-MiniLM-L6-v2
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5-coder:3b
OLLAMA_TIMEOUT_SECONDS=300
OLLAMA_KEEP_ALIVE=10m
OLLAMA_NUM_CTX=
OLLAMA_NUM_PREDICT=600
OLLAMA_PRE_WARM_ENABLED=true
OLLAMA_PRE_WARM_PROMPT=OK
OLLAMA_THINK=false
PROMPT_COMPRESSION_ENABLED=true
SCHEMA_CONTEXT_CACHE_ENABLED=true
SCHEMA_CONTEXT_CACHE_PATH=.cache/schema_context
INFERENCE_CACHE_ENABLED=true
INFERENCE_CACHE_PATH=.cache/inference
```

`OLLAMA_TIMEOUT_SECONDS=300` is only the maximum time the request is allowed to run. It does not make Ollama slower by itself. The slow part is usually local model generation, especially when the prompt asks for a complete response shape.

## Schema

The bundled schema is intentionally small:

```graphql
type Query {
  countries: [Country!]!
  country(code: ID!): Country
  continents: [Continent!]!
  continent(code: ID!): Continent
}

type Country {
  code: ID!
  name: String!
  native: String!
  emoji: String!
  capital: String
  currency: String
  continent: Continent!
  languages: [Language!]!
}

type Continent {
  code: ID!
  name: String!
}

type Language {
  code: ID!
  name: String
}
```

To use a different schema, update `resources/schema.graphql` or set `GRAPHQL_SCHEMA_FILE`:

```bash
GRAPHQL_SCHEMA_FILE=resources/other-schema.graphql .venv/bin/python -m graphql_ai.cli "Generate a sample query"
```

## PyCharm GraphQL Support

Install the JetBrains **GraphQL** plugin in PyCharm, then restart the IDE.

This project includes:

```text
graphql.config.yml
```

The GraphQL config points GraphQL tooling at `resources/schema.graphql`.
