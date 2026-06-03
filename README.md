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
