# Project Guidelines

This repository is a GraphQL AI examples project. The current implementation uses RAG to generate sample GraphQL operations, but future enhancements may add agents, inference optimization, model routing, prompt evaluation, or other approaches.

## Architecture

Keep code organized by responsibility:

```text
graphql_ai/
  api/       FastAPI routes and HTTP schemas
  core/      settings, protocols, shared framework utilities
  domain/    domain dataclasses and business objects
  llm/       LLM client protocols and provider implementations
  rag/       RAG-specific schema indexing and retrieval
  services/  business use cases and orchestration
  cli.py     command-line entry point
  main.py    FastAPI app factory
tests/
  api/       tests for API routes and HTTP behavior
  core/      tests for settings, protocols, and shared framework utilities
  domain/    tests for domain dataclasses and business objects
  llm/       tests for LLM clients and caching
  rag/       tests for schema chunking, indexing, and retrieval helpers
  services/  tests for business use cases and orchestration
```

Do not add a generic `utils` module or broad helper class unless the code is genuinely reused across multiple packages. Keep single-use helpers close to the service or module that owns the behavior.

## Layering Rules

- API routes must stay thin: validate/translate HTTP input, call a service, and return a Pydantic response.
- Services own business logic and orchestration. They may depend on protocols from `core/`, clients from `llm/`, and implementation modules such as `rag/`.
- Domain models must not depend on FastAPI, Chroma, Ollama, or framework code.
- RAG code must stay in `graphql_ai/rag`. Do not make RAG the identity of the whole application.
- Future approaches such as agents or inference optimization should be added as normal packages when implemented, not pre-created speculatively.
- LLM providers should implement the `LLMClient` protocol instead of being called directly from routes.
- Schema context providers should implement `SchemaContextProvider` so services can swap RAG for another approach later.

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

- Add docstrings for public classes and public methods.
- Public service classes should describe the business workflow they coordinate.
- When a service uses RAG, mention that RAG is the current schema-context approach, not the only possible approach.
- When a service applies validation, safety checks, or output filtering, describe the guardrail behavior clearly.
- Keep private helper docstrings optional; add them only when the behavior is not obvious.

## Testing

- Keep tests outside production code under `tests/`.
- Mirror the production package layout under `tests/`. For example, test `graphql_ai/services/sample_query_service.py` in `tests/services/test_sample_query_service.py`.
- Keep top-level tests only for top-level modules such as `graphql_ai/cli.py`.
- Prefer deterministic tests with fake LLM clients, fake schema-context providers, fake Chroma collections, and temporary files.
- Do not require live Ollama, Chroma, downloaded embedding models, or network access for the default test suite.
- Integration tests should cover application boundaries, such as FastAPI routes plus service wiring, while still replacing slow external AI dependencies with fakes.
- Live integration tests against real local infrastructure are optional and must be guarded behind an explicit environment variable such as `RUN_LIVE_INTEGRATION_TESTS=true`.
- When adding behavior, add or update the closest unit or integration test in the matching mirrored folder.

## RAG And Schema Handling

- `resources/schema.graphql` is the default schema and is expected to change rarely.
- Preserve Chroma index caching; avoid rebuilding the index per request.
- If schema indexing behavior changes, keep cache invalidation based on schema content and embedding model.
- Do not hard-code schema-specific fields outside a default example request unless the user explicitly asks for a schema-specific demo.

## Guardrails

- Validate generated GraphQL operations with GraphQL-core before returning them.
- Keep output guardrails in the service layer, not in API route handlers.
- Guardrails should reject malformed operations, invented fields, missing required arguments, invalid nested selections, and variable type mismatches.
- Keep variable-usage validation so Variables JSON cannot drift away from the generated operation.
- When changing guardrail behavior, add focused tests in `tests/services/` or the closest matching mirrored folder.

## Inference Optimization

- Keep inference optimization in the LLM or service layer, not in API route handlers.
- Preserve the local prompt/response cache unless the change explicitly replaces it.
- Preserve schema-context caching unless the change explicitly replaces request retrieval behavior.
- Prompt cache keys must include the full prompt plus model settings that affect output.
- Schema-context cache keys must include the user request and schema fingerprint.
- Keep prompt compression enabled by default for local inference; include compression settings in cache keys.
- Keep startup model pre-warming in the service/lifespan path, never in route handlers.
- Store runtime cache artifacts under ignored paths such as `.cache/`.
- Prefer small composable wrappers, such as cached LLM clients, over special cases inside business methods.

## Verification

Before finishing code changes, run the closest practical checks:

```bash
python3 -m py_compile graphql_ai/*.py graphql_ai/api/*.py graphql_ai/core/*.py graphql_ai/domain/*.py graphql_ai/llm/*.py graphql_ai/rag/*.py graphql_ai/services/*.py
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
