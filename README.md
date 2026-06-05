# GraphQL AI Examples

This is a small app for generating sample GraphQL calls and troubleshooting GraphQL calls from a local schema.

Sample generation uses RAG with Ollama or OpenAI inference. For `/sample/{root_field}`, the root field is converted to a focused prompt request, then retrieval provides schema context before prompt construction, inference, and guardrails. Troubleshooting uses a small tool-using agent. For `/troubleshoot/{root_field}`, the agent goal is to explain a submitted GraphQL call, run a plan with local tools, and suggest a corrected operation. In this app, `root_field` is the GraphQL Query or Mutation field name the user wants to generate or troubleshoot, such as `country`. The structure is intentionally open for adding other GraphQL AI capabilities later, such as planning workflows, inference optimization, model routing, and prompt evaluation.

## AI Concepts Covered

This project highlights common AI application patterns:

- **RAG**: retrieves schema context before generation.
- **Embeddings**: converts GraphQL SDL chunks into vectors.
- **Vector store**: persists schema embeddings in Chroma.
- **Retrieval**: selects schema chunks relevant to the requested root field.
- **Prompt construction**: combines system instructions, retrieved context, and the root-field request.
- **Prompt compression**: keeps retrieved schema context compact for local inference.
- **Inference**: sends the final prompt to a configured LLM provider.
- **Inference cache**: reuses responses for identical prompts and model settings.
- **Model pre-warm**: loads the local model during API startup to reduce first-request latency.
- **Guardrails**: validates input and generated GraphQL before returning it.
- **Agent**: coordinates a goal, plan, tool calls, observations, and inference for troubleshooting.
- **Plan**: ordered steps the agent follows to troubleshoot a GraphQL call.
- **Tools**: deterministic helpers for input guardrails, GraphQL validation, and schema retrieval.
- **Tool observations**: syntax errors, schema validation errors, and retrieved schema context passed into inference.
- **Prompt evaluation**: runs fixed cases and scores model output with existing GraphQL guardrails.
- **Model routing**: future extension point for selecting providers or models per workflow.

The RAG pipeline:

1. Reads `resources/schema.graphql`.
2. Splits the SDL into schema chunks.
3. Embeds those chunks with a local sentence-transformers model.
4. Stores the embeddings in a local Chroma index.
5. Runs retrieval to select schema context for the requested root field.
6. Builds the final prompt from system instructions, schema context, and the root-field request.
7. Checks the local inference cache for the final prompt.
8. Sends only uncached prompt context to the configured LLM provider for inference.
9. Applies GraphQL guardrails and returns the GraphQL operation plus Variables JSON.

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
.venv/bin/python -m graphql_ai.cli country
```

Example root field:

```bash
.venv/bin/python -m graphql_ai.cli countries
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

`/sample/{root_field}` calls always use the root-field path. The path value is the Query or Mutation field name from the schema that the user wants to generate. For the bundled schema, valid examples include `country`, `countries`, `continent`, and `continents`. The service converts that root field into a focused prompt request, then uses RAG, configured inference, and guardrails to generate the sample.

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
  "root_field": "country",
  "status": "invalid",
  "issues": [
    "Cannot query field 'code1' on type 'Country'. Did you mean 'code'? Location: line 3, column 5."
  ],
  "detail": [
    "Use the schema field `code` instead of `code1`."
  ],
  "suggestion": [
    "query CountryQuery($code: ID!) {",
    "  country(code: $code) {",
    "    code",
    "    name",
    "  }",
    "}"
  ]
}
```

When the submitted GraphQL operation is valid, troubleshooting returns empty issue and guidance fields:

```json
{
  "root_field": "country",
  "status": "valid",
  "issues": [],
  "detail": [],
  "suggestion": []
}
```

`/troubleshoot/{root_field}` uses a small agent workflow:

1. Goal: explain what is wrong with the submitted GraphQL operation and suggest a corrected operation.
2. Plan: validate input, parse and validate GraphQL, retrieve schema context, generate guidance, and validate the correction.
3. Tools: input guardrail, GraphQL-core validation, and RAG schema retrieval.
4. Tool observations: syntax locations, schema validation errors, and retrieved schema context.
5. Inference: the configured LLM provider receives the observations and proposes detail text plus a suggested operation.
6. Guardrail: the suggested operation is validated before it is returned. If it is still invalid, the API returns an empty `suggestion` and includes the validation issue.

Troubleshooting response fields come from different parts of the agent workflow:

- `issues`: GraphQL-core validation output.
- `detail`: model explanation for invalid calls, generated in the same inference call as the suggested operation.
- `suggestion`: model suggested operation, validated again with GraphQL-core before returning.

## Tests

Run the test suite:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

The tests use fake LLM and schema-context providers, so they do not require Chroma, the embedding model, a running Ollama server, or an OpenAI API key.

## Prompt Evaluation

Run the simple prompt evaluation suite:

```bash
source .venv/bin/activate
.venv/bin/python -m graphql_ai.evaluation.prompt_eval
```

Prompt evaluation runs a few fixed educational cases through the configured LLM provider and scores the results with existing guardrails:

- Sample generation output must validate against the GraphQL schema.
- Variables JSON must match variables used by the operation.
- Troubleshooting must return detail text and a suggested operation.
- Troubleshooting suggestions must validate against the GraphQL schema.
- Generated text must include a few expected root-field or response-field strings.

To evaluate only one workflow:

```bash
.venv/bin/python -m graphql_ai.evaluation.prompt_eval --workflow sample
.venv/bin/python -m graphql_ai.evaluation.prompt_eval --workflow troubleshoot
```

This is intentionally small. It is meant to show the prompt evaluation concept before adding richer datasets, prompt variants, model comparisons, or routing rules.

## Guardrails

The app applies input and output guardrails:

- Root-field input must match GraphQL field-name syntax before RAG retrieval or inference.
- GraphQL-core parses and validates generated operations against `resources/schema.graphql`.
- Invalid output is rejected before the API returns it, including malformed GraphQL, invented fields, missing required arguments, scalar fields with nested selections, and variable type mismatches.
- A separate variable-usage check rejects Variables JSON entries that are not referenced by the GraphQL operation.
- The troubleshooting agent captures syntax errors with line and column locations before asking the model for guidance.
- The troubleshooting agent validates its suggested operation before returning it.

These checks keep LLM output aligned with the schema and the API response contract.

## Inference Optimization

The app includes inference and retrieval optimizations:

- Chroma schema index cache: avoids re-embedding `resources/schema.graphql` on every run.
- Top-k schema retrieval: retrieves the most relevant schema chunks for `/sample/{root_field}`.
- Schema-context cache: avoids re-embedding and querying Chroma for repeated root-field requests.
- Troubleshooting schema cache: keeps the parsed GraphQL schema in memory for repeated validation.
- Troubleshooting retrieval cache: reuses retrieved schema context for repeated `/troubleshoot/{root_field}` calls.
- Inference response cache: avoids calling the LLM provider again when the final prompt and model settings are identical.
- Prompt compression: reduces prompt tokens by compacting schema context.
- Model pre-warm: loads the local model during FastAPI startup.

The inference cache is useful because generation is usually the slowest step. It is keyed by the full prompt plus provider settings and `PROMPT_CONTRACT_VERSION`, so changing the root-field request, retrieved schema context, model, Ollama runtime options, OpenAI output-token settings, or prompt contract produces a different cache entry.

Ollama runtime options are also tuned for local responsiveness:

- `OLLAMA_KEEP_ALIVE=10m` keeps the model loaded between requests.
- `OLLAMA_NUM_CTX` optionally controls the context window size.
- `OLLAMA_NUM_PREDICT=600` keeps the maximum output smaller for this simple schema.
- `OLLAMA_TEMPERATURE=0`, `OLLAMA_TOP_P=0.1`, `OLLAMA_TOP_K=1`, and `OLLAMA_SEED=42` reduce creative variance for schema-bound output.
- `PROMPT_COMPRESSION_ENABLED=true` keeps schema context and instructions compact before inference.
- `OLLAMA_PRE_WARM_ENABLED=true` sends a tiny request during FastAPI startup so the model is loaded before the first endpoint call.

Defaults:

```bash
LLM_PROVIDER=ollama
OLLAMA_KEEP_ALIVE=10m
OLLAMA_NUM_CTX=
OLLAMA_NUM_PREDICT=600
OLLAMA_TEMPERATURE=0
OLLAMA_TOP_P=0.1
OLLAMA_TOP_K=1
OLLAMA_SEED=42
OLLAMA_PRE_WARM_ENABLED=true
OLLAMA_PRE_WARM_PROMPT=OK
OPENAI_API_KEY=
OPENAI_URL=https://api.openai.com/v1/responses
OPENAI_MODEL=gpt-5.2
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_OUTPUT_TOKENS=600
PROMPT_COMPRESSION_ENABLED=true
PROMPT_CONTRACT_VERSION=29
SCHEMA_CONTEXT_CACHE_ENABLED=true
SCHEMA_CONTEXT_CACHE_PATH=.cache/schema_context
SCHEMA_CONTEXT_TOP_K=5
INFERENCE_CACHE_ENABLED=true
INFERENCE_CACHE_PATH=.cache/inference
```

To disable response caching:

```bash
INFERENCE_CACHE_ENABLED=false uvicorn graphql_ai.main:app --host 0.0.0.0 --port 8080
```

To disable model pre-warming:

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
  agents/
    troubleshooting_agent.py # Tool-using agent for GraphQL troubleshooting
  api/
    routes.py          # FastAPI controllers/routes
    schemas.py         # Pydantic request/response models
  core/
    config.py          # Environment-backed application settings
    protocols.py       # Protocol contracts used by services
    responses.py       # Shared response formatting
  domain/
    models.py          # Domain dataclasses shared by services and RAG
  evaluation/
    prompt_eval.py     # Simple prompt evaluation runner
  llm/
    base.py            # LLM client protocol
    cache.py           # Prompt/response cache wrapper
    factory.py         # Provider selection and cache wrapping
    ollama_client.py   # Ollama HTTP client
    openai_client.py   # OpenAI Responses API client
  rag/
    embeddings.py      # Local embedding model loading
    schema_chunks.py   # GraphQL SDL parsing/chunking
    vector_store.py    # Chroma indexing and retrieval
  services/
    sample_query_service.py # RAG and inference service for sample-query generation
  cli.py               # Command-line entry point
  main.py              # FastAPI app factory
tests/                 # Unit and integration tests with fake AI dependencies
  agents/
  api/
  core/
  domain/
  evaluation/
  llm/
  rag/
  services/
  test_cli.py
```

Design notes:

- API routes stay thin and delegate work to the service layer.
- Pydantic schemas define the public HTTP response contract.
- The troubleshooting agent owns its goal, plan, tools, tool observations, and final inference step.
- RAG is represented by the `graphql_ai/rag` module, but it is only the current schema-context approach.
- The sample-query service depends on a schema-context protocol, so RAG can be swapped or composed with another approach.
- LLM provider access is isolated behind client classes and an LLM protocol.
- GraphQL-core validation is used as an output guardrail before generated samples are returned.
- Prompt construction and output validation stay in the service layer.
- Retrieval, embeddings, and vector-store concerns stay in the RAG layer.
- Inference caching and model runtime options stay in the LLM layer.
- Application settings are centralized in `graphql_ai/core/config.py`.
- The Chroma collection is initialized once when the AI service starts instead of being rebuilt per request.
- Generation is serialized with a lock because schema retrieval and inference are expensive shared resources.

## Defaults

```bash
GRAPHQL_SCHEMA_FILE=resources/schema.graphql
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=graphql_schema
SCHEMA_CONTEXT_TOP_K=5
EMBEDDING_MODEL=resources/models/all-MiniLM-L6-v2
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5-coder:3b
OLLAMA_TIMEOUT_SECONDS=300
OLLAMA_KEEP_ALIVE=10m
OLLAMA_NUM_CTX=
OLLAMA_NUM_PREDICT=600
OLLAMA_TEMPERATURE=0
OLLAMA_TOP_P=0.1
OLLAMA_TOP_K=1
OLLAMA_SEED=42
OLLAMA_PRE_WARM_ENABLED=true
OLLAMA_PRE_WARM_PROMPT=OK
OLLAMA_THINK=false
OPENAI_API_KEY=
OPENAI_URL=https://api.openai.com/v1/responses
OPENAI_MODEL=gpt-5.2
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_OUTPUT_TOKENS=600
PROMPT_COMPRESSION_ENABLED=true
PROMPT_CONTRACT_VERSION=29
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
