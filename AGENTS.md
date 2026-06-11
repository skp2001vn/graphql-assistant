# Project Guidelines

This project generates and troubleshoots GraphQL operations from schema files.

## Architecture

```text
graphql_assistant/
  agents/
    tools/
  api/
  core/
  domain/
  llm/
  rag/
tests/
```

## Schema

- GraphQL schemas are loaded from `resources/*.graphql`.
- Generate and validate operations against the provided schema.
- Do not hardcode schema-specific fields in application logic.

## Layering Rules

- Routes stay thin.
- Agents belong in `graphql_assistant/agents`.
- Assistant tools own GraphQL assistant workflows.
- RAG belongs in `graphql_assistant/rag`.
- Prefer interfaces when multiple implementations are expected.
- Do not create packages for future features.

## Engineering Principles

- Simplicity first. Make every change as simple as possible.
- Verify generated and corrected GraphQL operations against the schema before returning them.
- Prefer deterministic validation over model assumptions.
- Fix root causes instead of symptoms.
- Keep changes small and localized.
- Reuse existing services before introducing new abstractions.
- Document the reasoning behind non-trivial design decisions.

## Naming

- Use clear module names.
- Use `Agent`, `Tool`, `Client`, and `Provider` suffixes consistently.
- Keep API schemas in `api/schemas.py`.
- Keep domain models in `domain/models.py`.

## FastAPI

- Create the app in `graphql_assistant/main.py`.
- Register routes in `graphql_assistant/api/routes.py`.
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
uvicorn graphql_assistant.main:app --host 127.0.0.1 --port 8082
curl -X POST http://127.0.0.1:8082/assistant \
  -H "Content-Type: application/json" \
  --data '{"goal": "Generate a sample query", "root_field": "country"}'
curl -X POST http://127.0.0.1:8082/assistant \
  -H "Content-Type: application/json" \
  --data '{"goal": "Troubleshoot this GraphQL operation", "root_field": "country", "graphql_call": "query CountryQuery($code: ID!) { country(code) { code name native emoji capital currency continent { code name } languages { code name } } }"}'
```
```
