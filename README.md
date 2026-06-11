# GraphQL Assistant

Generates and troubleshoots GraphQL query/mutation from local schema `*.graphql` files located in the `resources/` folder.

The application exposes one assistant surface, `/assistant`, and supports workflows:

- generate a sample GraphQL call for a query/mutation
- troubleshoot a submitted GraphQL query/mutation against the active schema
- generate mock data and documentation: future extension.

The application supports OpenAI API and local Ollama (by default).

## AI Concepts Covered 

- RAG: retrieves schema context before generation.
- Embeddings: converts GraphQL SDL chunks into vectors.
- Vector store: persists schema embeddings in Chroma.
- Retrieval: selects schema chunks relevant to the requested root field.
- Prompt construction: combines system instructions, retrieved context, and the root-field request.
- Prompt compression: keeps retrieved schema context compact for local inference.
- Inference: sends the final prompt to a configured LLM provider.
- Inference cache: reuses responses for identical prompts and model settings.
- Model pre-warm: loads the local model during API startup to reduce first-request latency.
- Guardrails: validates input and generated GraphQL before returning it.
- Agent: coordinates a goal, planner decision, tool execution, and inference through the unified assistant surface.
- Plan: the assistant reduces the user request to a structured intent such as `generate_sample`, `troubleshoot`, or `unsupported`.
- Tools: deterministic helpers and focused workflows for sample generation, GraphQL validation, and schema-aware troubleshooting.
- Prompt evaluation: runs fixed assistant cases and scores model output with existing GraphQL guardrails.

## RAG Pipeline

1. read `resources/schema.graphql`
2. split the SDL into schema chunks
3. embed those chunks with a local `sentence-transformers` model
4. store the embeddings in a local Chroma index
5. run retrieval to select schema context for the requested root field
6. build the final prompt from system instructions, schema context, and the root-field request
7. check the local inference cache for the final prompt
8. send only uncached prompt context to the configured LLM provider for inference
9. apply GraphQL guardrails and return the GraphQL operation plus Variables JSON

Because `resources/schema.graphql` is rarely updated, the Chroma index is cached and reused after the first run.

## Assistant Flow

The assistant layer is intentionally small:

- `GraphQLAssistantAgent` normalizes the request and asks Agno to classify the goal
- `SampleTool` handles sample generation
- `TroubleshootingTool` handles validation-aware troubleshooting and corrective suggestion

Agno is used only for structured workflow planning.

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

## CLI

Generate a sample operation:

```bash
source .venv/bin/activate
.venv/bin/python -m graphql_assistant.cli "Generate a sample query" country
```

Force a schema-index rebuild:

```bash
.venv/bin/python -m graphql_assistant.cli --rebuild "Generate a sample query" country
```

Troubleshoot a GraphQL operation:

```bash
.venv/bin/python -m graphql_assistant.cli "Troubleshoot this operation" country \
  --graphql-call 'query CountryQuery($code: ID!) { country(code: $code) { code name native emoji1 capital currency continent { code name } languages { code name } } }'
```

Use OpenAI instead of Ollama:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=your-api-key OPENAI_MODEL=gpt-5.4 \
.venv/bin/python -m graphql_assistant.cli "Troubleshoot this operation" country \
  --graphql-call 'query CountryQuery($code: ID!) { country(code: $code) { code name native emoji1 capital currency continent { code name } languages { code name } } }'

```

## API

Start the API:

```bash
source .venv/bin/activate
.venv/bin/uvicorn graphql_assistant.main:app --host 0.0.0.0 --port 8080
```

Generate a sample operation:

```bash
curl -X POST http://localhost:8080/assistant \
  -H "Content-Type: application/json" \
  --data '{
    "goal": "Generate a sample query",
    "root_field": "country"
  }'
```

`root_field` is the GraphQL Query or Mutation field to generate, such as `country` or `countries`.

Troubleshoot a GraphQL operation:

```bash
curl -X POST http://localhost:8080/assistant \
  -H "Content-Type: application/json" \
  --data '{
    "goal": "Troubleshoot this GraphQL operation",
    "root_field": "country",
    "graphql_call": "query CountryQuery($code: ID!) { country(code: $code) { code name native emoji1 capital currency continent { code name } languages { code name } } }"
  }'
```

When the submitted GraphQL operation is already valid, the troubleshooting workflow returns `status: "valid"` with empty `issues`, `detail`, and `suggestion` fields.

The main response fields are:

- `type`: `sample` or `troubleshooting`
- `operation`: generated GraphQL operation lines for sample generation
- `variables`: generated Variables JSON for sample generation
- `root_field`: root field under troubleshooting
- `status`: `valid` or `invalid` for troubleshooting
- `issues`: deterministic GraphQL validation issues
- `detail`: short user-facing explanation of the correction
- `suggestion`: corrected GraphQL operation lines when the tool can produce a valid fix

## Prompt Evaluation

Run assistant-level prompt evals:

```bash
.venv/bin/python -m graphql_assistant.evaluation.prompt_eval
```

Filter by assistant intent:

```bash
.venv/bin/python -m graphql_assistant.evaluation.prompt_eval --intent generate_sample
.venv/bin/python -m graphql_assistant.evaluation.prompt_eval --intent troubleshoot
.venv/bin/python -m graphql_assistant.evaluation.prompt_eval --intent unsupported
```

The eval runner exercises the public assistant flow and scores outputs with the same schema guardrails used by the app.

## Tests

Run the test suite:

```bash
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
```

The tests use fake LLM and schema-context providers, so they do not require Chroma, the embedding model, a running Ollama server, or an OpenAI API key.

## Project Structure

```text
graphql_assistant/
  agents/      assistant planner and tools
  api/         FastAPI routes and schemas
  core/        shared settings and interfaces
  domain/      domain models
  llm/         LLM clients, caching, and Agno adapter
  rag/         schema chunking, embeddings, and vector retrieval
  evaluation/  prompt eval runner
tests/
resources/
```

## Design Notes

- Routes stay thin.
- The assistant owns workflow selection.
- Assistant tools own GraphQL workflow logic.
- RAG concerns stay in `rag/`.
- LLM access stays in `llm/`.
