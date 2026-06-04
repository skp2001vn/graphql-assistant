from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from graphql_ai.api.routes import router
from graphql_ai.core.responses import PrettyJSONResponse
from graphql_ai.domain import GeneratedGraphQLSample
from graphql_ai.main import create_app


class FakeSampleService:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.root_fields: list[str] = []
        self.pre_warm_called = False

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        self.root_fields.append(root_field)
        if self.fail:
            raise RuntimeError("generation failed")

        return GeneratedGraphQLSample(
            operation="query CountryQuery($code: ID!) {\n  country(code: $code) {\n    code\n  }\n}",
            variables={"code": "US"},
            raw_response="raw",
        )

    def pre_warm(self) -> None:
        self.pre_warm_called = True


def build_test_client(sample_service: FakeSampleService) -> TestClient:
    app = FastAPI(default_response_class=PrettyJSONResponse)
    app.include_router(router)
    app.state.sample_service = sample_service
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

    def test_sample_endpoint_returns_503_when_generation_fails(self) -> None:
        client = build_test_client(FakeSampleService(fail=True))

        response = client.get("/sample/country")

        self.assertEqual(503, response.status_code)
        self.assertEqual({"detail": "generation failed"}, response.json())

    def test_create_app_lifespan_constructs_and_prewarm_service(self) -> None:
        fake_service = FakeSampleService()

        with patch("graphql_ai.main.SampleQueryService", return_value=fake_service) as service_class:
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/health")

        self.assertEqual(200, response.status_code)
        service_class.assert_called_once()
        self.assertTrue(fake_service.pre_warm_called)

    def test_create_app_uses_startup_service_for_sample_request(self) -> None:
        fake_service = FakeSampleService()

        with patch("graphql_ai.main.SampleQueryService", return_value=fake_service) as service_class:
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/sample/country")

        self.assertEqual(200, response.status_code)
        service_class.assert_called_once()
        self.assertEqual(["country"], fake_service.root_fields)


if __name__ == "__main__":
    unittest.main()
