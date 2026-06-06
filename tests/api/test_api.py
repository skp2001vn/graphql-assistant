from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from graphql_ai.agents import AgentPlanningError, GraphQLAIGoal, GraphQLAIResult
from graphql_ai.api.routes import router
from graphql_ai.core.responses import PrettyJSONResponse
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_ai.main import create_app


class FakeLLMPreWarmer:
    def __init__(self, settings: object, llm_client: object) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.pre_warm_called = False

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeSampleService:
    def __init__(self) -> None:
        self.pre_warm_called = False

    def pre_warm(self) -> None:
        self.pre_warm_called = True


class FakeTroubleshootingService:
    pass


class FakeGraphQLAIAgent:
    def __init__(self, error: Exception | None = None, output: object | None = None) -> None:
        self.error = error
        self.output = output
        self.goals: list[GraphQLAIGoal] = []

    def run(self, goal: GraphQLAIGoal) -> GraphQLAIResult:
        self.goals.append(goal)
        if self.error is not None:
            raise self.error

        output = self.output or GeneratedGraphQLSample(
            operation="query CountryQuery($code: ID!) {\n  country(code: $code) {\n    code\n  }\n}",
            variables={"code": "US"},
            raw_response="raw",
        )
        intent = "troubleshoot" if isinstance(output, TroubleshootingResult) else "generate_sample"
        return GraphQLAIResult(
            intent=intent,
            goal=goal,
            plan=(),
            tool_calls=(),
            observations=(),
            output=output,
            raw_plan_response="{}",
        )


def build_test_client(graphql_ai_agent: FakeGraphQLAIAgent) -> TestClient:
    app = FastAPI(default_response_class=PrettyJSONResponse)
    app.include_router(router)
    app.state.graphql_ai_agent = graphql_ai_agent
    return TestClient(app)


class ApiTest(unittest.TestCase):
    def test_health_endpoint_returns_status(self) -> None:
        client = build_test_client(FakeGraphQLAIAgent())

        response = client.get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.json())

    def test_assistant_returns_sample_result(self) -> None:
        agent = FakeGraphQLAIAgent()
        client = build_test_client(agent)

        response = client.post(
            "/assistant",
            json={
                "goal": "Generate a sample query",
                "root_field": "country",
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "type": "sample",
                "operation": [
                    "query CountryQuery($code: ID!) {",
                    "  country(code: $code) {",
                    "    code",
                    "  }",
                    "}",
                ],
                "variables": {"code": "US"},
                "root_field": None,
                "status": None,
                "issues": None,
                "detail": None,
                "suggestion": None,
            },
            response.json(),
        )
        self.assertEqual([GraphQLAIGoal(goal="Generate a sample query", root_field="country")], agent.goals)

    def test_assistant_returns_troubleshooting_result(self) -> None:
        graphql_call = "query CountryQuery($code: ID!) { county(code: $code) { code } }"
        troubleshooting_result = TroubleshootingResult(
            root_field="country",
            status="invalid",
            issues=["Cannot query field 'county' on type 'Query'. Did you mean 'country'?"],
            detail=["Use the schema field `country` instead of `county`."],
            suggestion="query CountryQuery($code: ID!) {\n  country(code: $code) {\n    code\n  }\n}",
            raw_response="raw",
        )
        agent = FakeGraphQLAIAgent(output=troubleshooting_result)
        client = build_test_client(agent)

        response = client.post(
            "/assistant",
            json={
                "goal": "Something is wrong with this query",
                "root_field": "country",
                "graphql_call": graphql_call,
            },
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "type": "troubleshooting",
                "operation": None,
                "variables": None,
                "root_field": "country",
                "status": "invalid",
                "issues": ["Cannot query field 'county' on type 'Query'. Did you mean 'country'?"],
                "detail": ["Use the schema field `country` instead of `county`."],
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
        self.assertEqual(
            [GraphQLAIGoal(goal="Something is wrong with this query", root_field="country", graphql_call=graphql_call)],
            agent.goals,
        )

    def test_assistant_returns_400_for_agent_planning_error(self) -> None:
        client = build_test_client(
            FakeGraphQLAIAgent(
                error=AgentPlanningError(
                    "Troubleshooting requires `graphql_call`. Include the GraphQL operation in the request body."
                )
            )
        )

        response = client.post(
            "/assistant",
            json={
                "goal": "Something is wrong with this query",
                "root_field": "country",
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual(
            {
                "detail": (
                    "Troubleshooting requires `graphql_call`. "
                    "Include the GraphQL operation in the request body."
                )
            },
            response.json(),
        )

    def test_sample_and_troubleshoot_routes_are_removed(self) -> None:
        client = build_test_client(FakeGraphQLAIAgent())

        sample_response = client.get("/sample/country")
        troubleshoot_response = client.post("/troubleshoot/country", content="query CountryQuery { country { code } }")

        self.assertEqual(404, sample_response.status_code)
        self.assertEqual(404, troubleshoot_response.status_code)

    def test_create_app_lifespan_constructs_services_and_assistant_agent(self) -> None:
        sample_service = FakeSampleService()
        troubleshooting_service = FakeTroubleshootingService()
        settings = object()
        schema_context_provider = object()
        llm_client = object()
        pre_warmer = FakeLLMPreWarmer(settings, llm_client)

        with (
            patch("graphql_ai.main.get_settings", return_value=settings),
            patch("graphql_ai.main.SchemaVectorStore", return_value=schema_context_provider) as vector_store_class,
            patch("graphql_ai.main.build_llm_client", return_value=llm_client) as llm_factory,
            patch("graphql_ai.main.LLMPreWarmer", return_value=pre_warmer) as pre_warmer_class,
            patch("graphql_ai.main.SampleQueryService", return_value=sample_service) as sample_service_class,
            patch(
                "graphql_ai.main.TroubleshootingService",
                return_value=troubleshooting_service,
            ) as troubleshooting_service_class,
            patch("graphql_ai.main.GraphQLAIAgent") as agent_class,
        ):
            app = create_app()
            with TestClient(app) as client:
                response = client.get("/health")

        self.assertEqual(200, response.status_code)
        vector_store_class.assert_called_once_with(settings=settings)
        llm_factory.assert_called_once_with(settings)
        pre_warmer_class.assert_called_once_with(settings, llm_client)
        self.assertTrue(pre_warmer.pre_warm_called)
        sample_service_class.assert_called_once_with(
            settings=settings,
            llm_client=llm_client,
            llm_pre_warmer=pre_warmer,
            schema_context_provider=schema_context_provider,
        )
        troubleshooting_service_class.assert_called_once_with(
            settings=settings,
            llm_client=llm_client,
            llm_pre_warmer=pre_warmer,
            schema_context_provider=schema_context_provider,
        )
        agent_class.assert_called_once_with(
            llm_client=llm_client,
            sample_query_tool=sample_service,
            troubleshooting_tool=troubleshooting_service,
        )
        self.assertFalse(sample_service.pre_warm_called)


if __name__ == "__main__":
    unittest.main()
