# Project Guidelines

This project generates and troubleshoots GraphQL operations from schema files.

## Architecture

```text
graphql_ai/
  agents/
  api/
  core/
  domain/
  llm/
  rag/
  services/
tests/
```

## Schema

- GraphQL schemas are loaded from `resources/*.graphql`.
- Generate and validate operations against the provided schema.
- Do not hardcode schema-specific fields in application logic.

## Layering Rules

- Routes stay thin.
- Services own business logic.
- Agents belong in `graphql_ai/agents`.
- RAG belongs in `graphql_ai/rag`.
- Prefer interfaces when multiple implementations are expected.
- Do not create packages for future features.

## Naming

- Use clear module names.
- Use `Service`, `Client`, and `Provider` suffixes consistently.
- Keep API schemas in `api/schemas.py`.
- Keep domain models in `domain/models.py`.

## FastAPI

- Create the app in `graphql_ai/main.py`.
- Register routes in `graphql_ai/api/routes.py`.
- Use lifespan for application initialization.
- Do not call LLM or RAG code directly from routes.

## Documentation

- Add docstrings for public classes and methods.
- Document business workflows when helpful.

## Testing

- Keep tests under `tests/`.
- Mirror the production package layout.
- Prefer deterministic tests with fakes.
- Do not require external services for default tests.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests
```

```md
Smoke test API changes when relevant:

```bash
uvicorn graphql_ai.main:app --host 127.0.0.1 --port 8082
curl http://127.0.0.1:8082/sample/country
curl -X POST http://127.0.0.1:8082/troubleshoot/country
```
```
