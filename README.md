# GraphQL AI Examples

A small example project for generating GraphQL operations and troubleshooting GraphQL queries from a local schema.

- Sample generation uses RAG.
- Troubleshooting uses a tool-using agent.
- Supports Ollama and OpenAI.

The application generates and validates GraphQL operations from schema files located in the resources/ folder (typically *.graphql files).

## Overview

This project demonstrates two AI workflows:

- Sample GraphQL generation using RAG.
- GraphQL troubleshooting using a small agent workflow.

Key concepts:

- RAG (retrieval-augmented generation)
- Embeddings and vector search
- LLM inference (Ollama or OpenAI)
- Output validation (guardrails)
- Tool-using agents
- GraphQL schema-driven generation and validation

## Setup

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
.venv/bin/python -m graphql_ai.cli country
```

The first run builds the local Chroma index. Later runs reuse it automatically. If you edit `resources/schema.graphql`, the app detects the schema change and rebuilds the index.

To force a rebuild:

```bash
.venv/bin/python -m graphql_ai.cli --rebuild country
```

To use OpenAI instead of Ollama:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=your-api-key OPENAI_MODEL=gpt-5.2 .venv/bin/python -m graphql_ai.cli country
```

## API

Start the API:

```bash
source .venv/bin/activate
uvicorn graphql_ai.main:app --host 0.0.0.0 --port 8080
```

Call the sample-query endpoint:

```bash
curl http://localhost:8080/sample/country
```

The response is JSON:

```json
{
  "operation": ["..."],
  "variables": {
    "code": "US"
  }
}
```

`root_field` is the GraphQL Query or Mutation field to generate, such as `country` or `countries`.

Call the troubleshooting endpoint with a plain-text GraphQL operation:

```bash
curl -X POST http://localhost:8080/troubleshoot/country \
  -H "Content-Type: text/plain" \
  --data 'query CountyQuery($code: ID!) {
  country(code: $code) {
    code1
    name
  }
}'
```

Postman's **Body > GraphQL** mode sends JSON with a `query` field. The endpoint accepts that shape too:

```json
{
  "query": "query CountryQuery($code: ID!) {\n  country(code: $code) {\n    code1\n    name\n  }\n}",
  "variables": {
    "code": "US"
  }
}
```

The response is JSON:

```json
{
  "status": "invalid",
  "issues": ["..."],
  "detail": ["..."],
  "suggestion": ["..."]
}
```

When the submitted GraphQL operation is valid, troubleshooting returns empty issue and guidance fields:

```json
{
  "status": "valid",
  "issues": [],
  "detail": [],
  "suggestion": []
}
```

The troubleshooting workflow validates the operation, retrieves relevant schema context, generates guidance, and returns a corrected operation when possible.

Response fields:

- `issues`: validation errors.
- `detail`: explanation of the problem.
- `suggestion`: corrected GraphQL operation.

## Tests

Run the test suite:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

The tests use fake LLM and schema-context providers, so they do not require Chroma, the embedding model, a running Ollama server, or an OpenAI API key.

## Project Structure

```text
graphql_ai/
  agents/    troubleshooting workflow
  api/       FastAPI routes and schemas
  core/      shared settings and interfaces
  domain/    domain models
  llm/       LLM providers
  rag/       schema retrieval
  services/  business workflows
tests/
```

Design principles:

- Routes stay thin.
- Services contain business logic.
- Agents contain agent workflows.
- RAG concerns stay in `rag/`.
- LLM access stays in `llm/`.
