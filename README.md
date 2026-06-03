# GraphQL Local RAG Demo

This is a small educational app for generating sample GraphQL calls from a local schema.

The app keeps RAG in the flow on purpose:

1. Reads `resources/schema.graphql`.
2. Splits the SDL into schema chunks.
3. Embeds those chunks with a local sentence-transformers model.
4. Stores the embeddings in a local Chroma index.
5. Retrieves schema context for your request.
6. Sends only that prompt context to local Ollama and prints the GraphQL operation plus Variables JSON.

Because `resources/schema.graphql` is rarely updated, the Chroma index is cached and reused after the first run.

## One-Time Setup

Run this once while online:

```bash
python -m venv .venv
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
python graphql_rag_local.py "Generate a sample query for a country by code"
```

Example request:

```bash
python graphql_rag_local.py "Generate a sample query for all countries"
```

The first run builds the local Chroma index. Later runs reuse it automatically. If you edit `resources/schema.graphql`, the app detects the schema change and rebuilds the index.

To force a rebuild:

```bash
python graphql_rag_local.py --rebuild "Generate a sample query for a country by code"
```

## Run The API

Start the FastAPI server:

```bash
source .venv/bin/activate
uvicorn graphql_rag_api:app --host 0.0.0.0 --port 8080
```

Call the sample-query endpoint:

```bash
curl http://localhost:8080/generatesamplequery/country
```

If port `8080` is already in use, either stop the existing server or use another port:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
kill 94081
```

Replace `94081` with the PID shown by `lsof`.

Or run the API on a different port:

```bash
uvicorn graphql_rag_api:app --host 0.0.0.0 --port 8081
curl http://localhost:8081/generatesamplequery/country
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
curl "http://localhost:8080/generatesamplequery/country?request=Generate%20a%20sample%20query%20for%20all%20countries"
```

The API follows the same RAG flow as the CLI. It builds or reuses the Chroma schema index once during application startup, then each endpoint call retrieves schema context and asks Ollama to generate the sample GraphQL call.

Design notes:

- `graphql_rag_local.py` owns the reusable RAG and Ollama generation logic.
- `graphql_rag_api.py` owns only the HTTP API layer.
- The API uses typed Pydantic response models.
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
OLLAMA_NUM_PREDICT=1200
OLLAMA_THINK=false
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
GRAPHQL_SCHEMA_FILE=resources/other-schema.graphql python graphql_rag_local.py "Generate a sample query"
```

## PyCharm GraphQL Support

Install the JetBrains **GraphQL** plugin in PyCharm, then restart the IDE.

This project includes:

```text
graphql.config.yml
```

The GraphQL config points GraphQL tooling at `resources/schema.graphql`.
