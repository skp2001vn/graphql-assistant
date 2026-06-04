from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from graphql_ai.api.routes import router
from graphql_ai.core.responses import PrettyJSONResponse
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_ai.main import create_app
from graphql_ai.services.sample_query_service import InvalidRootFieldNameError


class FakeSampleService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.root_fields: list[str] = []
        self.pre_warm_called = False
        self.settings = object()
        self.llm_client = object()
        self.schema_context_provider = object()

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        self.root_fields.append(root_field)
        if self.error is not None:
            raise self.error

        return GeneratedGraphQLSample(
            operation="query CountryQuery($code: ID!) {\n  country(code: $code) {\n    code\n  }\n}",
            variables={"code": "US"},
            raw_response="raw",
        )

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeTroubleshootingAgent:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[tuple[str, str]] = []

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        self.requests.append((root_field, graphql_call))
        if self.error is not None:
            raise self.error

        return TroubleshootingResult(
            root_field=root_field,
            status="invalid",
            issues=["Cannot query field 'county' on type 'Query'. Did you mean 'country'?"],
            detail="Use the schema field `country` instead of `county`.",
            suggestion="query CountryQuery($code: ID!) {\n  country(code: $code) {\n    code\n  }\n}",
            raw_response="raw",
        )


def build_test_client(
    sample_service: FakeSampleService,
    troubleshooting_agent: FakeTroubleshootingAgent | None = None,
) -> TestClient:
    app = FastAPI(default_response_class=PrettyJSONResponse)
    app.include_router(router)
    app.state.sample_service = sample_service
    app.state.troubleshooting_agent = troubleshooting_agent or FakeTroubleshootingAgent()
    return TestClient(app)


class ApiTest(unittest.TestCase):
    def test_health_endpoint_returns_status(self) -> None:
        client = build_test_client(FakeSampleService())

        response = client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.json())

    def test_sample_endpoint_returns_operation_lines_and_variables(self) -> None:
        service = FakeSampleService()
        client = build_test_client(service)

        response = client.get("/sample/country")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "operation": [
                    "query CountryQuery($code: ID!) {",
                    "  country(code: $code) {",
                    "    code",
                    "  }",
                    "}",
                ],
                "variables": {"code": "US"},
            },
            response.json(),
        )
        self.assertEqual(["country"], service.root_fields)

    def test_sample_endpoint_returns_400_when_root_field_name_is_invalid(self) -> None:
        client = build_test_client(FakeSampleService(InvalidRootFieldNameError("invalid root field")))

        response = client.get("/sample/invalid")

        self.assertEqual(400, response.status_code)
        self.assertEqual({"detail": "invalid root field"}, response.json())

    def test_sample_endpoint_returns_503_when_generation_fails(self) -> None:
        client = build_test_client(FakeSampleService(RuntimeError("generation failed")))

        response = client.get("/sample/country")

        self.assertEqual(503, response.status_code)
        self.assertEqual({"detail": "generation failed"}, response.json())

    def test_troubleshoot_endpoint_returns_agent_result(self) -> None:
        troubleshooting_agent = FakeTroubleshootingAgent()
        client = build_test_client(FakeSampleService(), troubleshooting_agent)
        graphql_call = 'query CountyQuery($code: ID!) { county(code: $code) { code } }'

        response = client.post(
            "/troubleshoot/county",
            content=graphql_call,
            headers={"Content-Type": "text/plain"},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "root_field": "county",
                "status": "invalid",
                "issues": ["Cannot query field 'county' on type 'Query'. Did you mean 'country'?"],
                "detail": "Use the schema field `country` instead of `county`.",
                "suggestion": [
                    "query CountryQuery($code: ID!) {",
                    "  country(code: $code) {",
                    "    code",
                    "  }",
                    "}",
                ],
            },
            response.json(),
        )
        self.assertEqual([("county", graphql_call)], troubleshooting_agent.requests)

    def test_troubleshoot_endpoint_accepts_postman_graphql_json_body(self) -> None:
        troubleshooting_agent = FakeTroubleshootingAgent()
        client = build_test_client(FakeSampleService(), troubleshooting_agent)
        graphql_call = 'query CountyQuery($code: ID!) { county(code: $code) { code } }'

        response = client.post(
            "/troubleshoot/county",
            json={
                "query": graphql_call,
                "variables": {"code": "US"},
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("invalid", response.json()["status"])
        self.assertEqual([("county", graphql_call)], troubleshooting_agent.requests)

    def test_troubleshoot_endpoint_rejects_json_body_without_query(self) -> None:
        client = build_test_client(FakeSampleService(), FakeTroubleshootingAgent())

        response = client.post("/troubleshoot/country", json={"variables": {"code": "US"}})

        self.assertEqual(400, response.status_code)
        self.assertEqual(
            {"detail": "Request JSON body must include a string `query` field."},
            response.json(),
        )

    def test_troubleshoot_endpoint_returns_400_for_invalid_root_field(self) -> None:
        client = build_test_client(
            FakeSampleService(),
            FakeTroubleshootingAgent(InvalidRootFieldNameError("invalid root field")),
        )

        response = client.post(
            "/troubleshoot/invalid",
            content="query CountryQuery { country(code: \"US\") { code } }",
            headers={"Content-Type": "text/plain"},
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual({"detail": "invalid root field"}, response.json())

    def test_create_app_lifespan_constructs_and_prewarm_service(self) -> None:
        fake_service = FakeSampleService()

        with (
            patch("graphql_ai.main.SampleQueryService", return_value=fake_service) as service_class,
            patch("graphql_ai.main.TroubleshootingAgent") as agent_class,
        ):
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/health")

        self.assertEqual(200, response.status_code)
        service_class.assert_called_once()
        agent_class.assert_called_once_with(
            settings=fake_service.settings,
            llm_client=fake_service.llm_client,
            schema_context_provider=fake_service.schema_context_provider,
        )
        self.assertTrue(fake_service.pre_warm_called)

    def test_create_app_uses_startup_service_for_sample_request(self) -> None:
        fake_service = FakeSampleService()

        with (
            patch("graphql_ai.main.SampleQueryService", return_value=fake_service) as service_class,
            patch("graphql_ai.main.TroubleshootingAgent"),
        ):
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/sample/country")

        self.assertEqual(200, response.status_code)
        service_class.assert_called_once()
        self.assertEqual(["country"], fake_service.root_fields)


if __name__ == "__main__":
    unittest.main()
