# Project Guidelines

This repository is a GraphQL AI examples project. The current implementation uses RAG to generate sample GraphQL operations and a tool-using agent to troubleshoot GraphQL calls. Future enhancements may add inference optimization, model routing, prompt evaluation, or other approaches.

## AI Terminology

Use consistent terminology throughout the project:

- RAG: retrieve schema context before generation.
- Retrieval: select relevant schema chunks for a request.
- Embeddings / Vector Store: schema indexing and search.
- Inference: LLM generation.
- Guardrails: validation and output checks.
- Agent: goal-driven workflow that may use tools.

## Architecture

```text
graphql_ai/
  agents/    agent workflows
  api/       FastAPI routes and schemas
  core/      shared interfaces and settings
  domain/    domain models
  llm/       LLM providers and caching
  rag/       schema retrieval and indexing
  services/  business workflows
tests/
  mirrors the production package layout
```

Do not add a generic `utils` module or broad helper class unless the code is genuinely reused across multiple packages. Keep single-use helpers close to the service or module that owns the behavior.

## Layering Rules

- API routes must stay thin: validate/translate HTTP input, call a service, and return a Pydantic response.
- Services own business logic and orchestration.
- Domain models must not depend on FastAPI, Chroma, Ollama, or framework code.
- Agent workflows belong in `graphql_ai/agents`.
- RAG code must stay in `graphql_ai/rag`. Do not make RAG the identity of the whole application.
- Do not create packages for features that do not exist yet.
- LLM providers should implement the `LLMClient` protocol instead of being called directly from routes.
- Prefer interfaces when multiple implementations are expected.

## Naming Conventions

- Use clear Python module names with nouns for packages and service names, for example `sample_query_service.py`.
- Use `Service` suffix only for business use-case orchestration classes.
- Use `Client` suffix for external service adapters such as Ollama.
- Use `Provider` suffix for protocol-like context sources.
- Keep API response models in `api/schemas.py`.
- Keep domain dataclasses in `domain/models.py`.

## FastAPI Conventions

- Create the FastAPI app in `graphql_ai/main.py`.
- Register routes through routers from `graphql_ai/api/routes.py`.
- Use FastAPI lifespan for application-scoped service initialization.
- Return typed Pydantic response models from route handlers.
- Keep JSON responses pretty-formatted for this project.
- Do not call Ollama, Chroma, or embedding code directly from route handlers.

## Documentation

- Add docstrings for public classes and methods.
- Describe business workflows in service classes.
- Keep terminology consistent with the AI Terminology section.
- Keep private helper documentation optional.

## Testing

- Keep tests outside production code under `tests/`.
- Mirror the production package layout under `tests/`. For example, test `graphql_ai/services/sample_query_service.py` in `tests/services/test_sample_query_service.py`.
- Keep top-level tests only for top-level modules such as `graphql_ai/cli.py`.
- Prefer deterministic tests with fake LLM clients, fake schema-context providers, fake Chroma collections, and temporary files.
- Do not require live Ollama, Chroma, downloaded embedding models, or network access for the default test suite.
- Integration tests should cover application boundaries, such as FastAPI routes plus service wiring, while still replacing slow external AI dependencies with fakes.
- Live integration tests against real local infrastructure are optional and must be guarded behind an explicit environment variable such as `RUN_LIVE_INTEGRATION_TESTS=true`.
- When adding behavior, add or update the closest unit or integration test in the matching mirrored folder.

## Verification

Before finishing code changes, run the closest practical checks:

```bash
python3 -m py_compile graphql_ai/*.py graphql_ai/agents/*.py graphql_ai/api/*.py graphql_ai/core/*.py graphql_ai/domain/*.py graphql_ai/llm/*.py graphql_ai/rag/*.py graphql_ai/services/*.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m graphql_ai.cli --help
```

For API changes, also smoke-test with Uvicorn on a free port:

```bash
uvicorn graphql_ai.main:app --host 127.0.0.1 --port 8082
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8082/sample/country
```

Use another port if `8082` is already in use.

## Git Hygiene

- Do not commit local runtime artifacts such as `.venv/`, `chroma_db/`, `resources/models/`, or `__pycache__/`.
- Keep changes scoped to the requested enhancement.
- Do not reintroduce the deleted top-level wrappers `graphql_rag_local.py` or `graphql_rag_api.py`; use `graphql_ai.cli` and `graphql_ai.main:app`.
