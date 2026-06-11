from __future__ import annotations

import unittest

from graphql_assistant.agents import AgentPlanningError, GraphQLAssistantAgent, GraphQLAssistantGoal
from graphql_assistant.agents.assistant_agent import AgnoAssistantPlanner
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeSampleTool:
    def __init__(self) -> None:
        self.root_fields: list[str] = []

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        self.root_fields.append(root_field)
        return GeneratedGraphQLSample(
            operation="query CountryQuery { country(code: \"US\") { code } }",
            variables={},
            raw_response="raw sample",
        )


class FakeTroubleshootingTool:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        self.requests.append((root_field, graphql_call))
        return TroubleshootingResult(
            root_field=root_field,
            status="invalid",
            issues=["Cannot query field 'county' on type 'Query'."],
            detail=["Use `country`."],
            suggestion="query CountryQuery { country(code: \"US\") { code } }",
            raw_response="raw troubleshoot",
        )


class FakePlanner:
    def __init__(self, intent: str, reason: str = "selected", raw_response: str = "{}") -> None:
        self.intent = intent
        self.reason = reason
        self.raw_response = raw_response
        self.goals: list[GraphQLAssistantGoal] = []

    def choose_intent(self, goal: GraphQLAssistantGoal) -> tuple[str, str, str]:
        self.goals.append(goal)
        return self.intent, self.reason, self.raw_response


class GraphQLAssistantAgentTest(unittest.TestCase):
    def test_agno_planner_uses_structured_output(self) -> None:
        llm_client = FakeLLMClient(
            '{"intent":"generate_sample","reason":"The user asked for a sample query."}'
        )

        intent, reason, raw_response = AgnoAssistantPlanner(llm_client).choose_intent(
            GraphQLAssistantGoal(goal="Generate a sample query", root_field="country")
        )

        self.assertIn("Generate a sample query", llm_client.prompts[0])
        self.assertEqual("generate_sample", intent)
        self.assertEqual("The user asked for a sample query.", reason)
        self.assertIn("generate_sample", raw_response)

    def test_agno_planner_can_return_unsupported_intent(self) -> None:
        llm_client = FakeLLMClient('{"intent":"unsupported","reason":"The goal is unclear."}')

        intent, reason, raw_response = AgnoAssistantPlanner(llm_client).choose_intent(
            GraphQLAssistantGoal(
                goal="sdfdsfdf",
                root_field="country",
                graphql_call="query CountryQuery { country { code } }",
            )
        )

        self.assertEqual("unsupported", intent)
        self.assertEqual("The goal is unclear.", reason)
        self.assertIn("unsupported", raw_response)

    def test_sample_goal_uses_plan_and_calls_sample_tool(self) -> None:
        planner = FakePlanner("generate_sample", "The user asked for a sample query.")
        sample_tool = FakeSampleTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAssistantAgent(FakeLLMClient("unused"), sample_tool, troubleshooting_tool, planner=planner)

        result = agent.run(GraphQLAssistantGoal(goal="Generate a sample query", root_field="country"))

        self.assertEqual([GraphQLAssistantGoal(goal="Generate a sample query", root_field="country")], planner.goals)
        self.assertEqual(["country"], sample_tool.root_fields)
        self.assertEqual([], troubleshooting_tool.requests)
        self.assertEqual("generate_sample", result.intent)
        self.assertIsInstance(result.output, GeneratedGraphQLSample)

    def test_troubleshoot_goal_uses_plan_and_calls_troubleshooting_tool(self) -> None:
        graphql_call = "query CountyQuery($code: ID!) { county(code: $code) { code } }"
        planner = FakePlanner("troubleshoot", "The user asked to fix a GraphQL operation.")
        sample_tool = FakeSampleTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAssistantAgent(FakeLLMClient("unused"), sample_tool, troubleshooting_tool, planner=planner)

        result = agent.run(
            GraphQLAssistantGoal(
                goal="Something is wrong with this query",
                root_field="country",
                graphql_call=graphql_call,
            )
        )

        self.assertEqual([], sample_tool.root_fields)
        self.assertEqual([("country", graphql_call)], troubleshooting_tool.requests)
        self.assertEqual("troubleshoot", result.intent)
        self.assertIsInstance(result.output, TroubleshootingResult)

    def test_graphql_call_does_not_force_troubleshooting_intent(self) -> None:
        graphql_call = "query CountryQuery($code: ID!) { country(code: $code) { code name native emoji1 capital currency continent { code name } languages { code name } } }"
        planner = FakePlanner("generate_sample", "The user asked for a fresh sample query.")
        sample_tool = FakeSampleTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAssistantAgent(FakeLLMClient("unused"), sample_tool, troubleshooting_tool, planner=planner)

        result = agent.run(
            GraphQLAssistantGoal(
                goal="Generate a sample query using this operation as context",
                root_field="country",
                graphql_call=graphql_call,
            )
        )

        self.assertEqual(["country"], sample_tool.root_fields)
        self.assertEqual([], troubleshooting_tool.requests)
        self.assertEqual("generate_sample", result.intent)

    def test_troubleshoot_plan_requires_request_graphql_call(self) -> None:
        planner = FakePlanner("troubleshoot", "The user asked to troubleshoot.")
        agent = GraphQLAssistantAgent(
            FakeLLMClient("unused"),
            FakeSampleTool(),
            FakeTroubleshootingTool(),
            planner=planner,
        )

        with self.assertRaisesRegex(AgentPlanningError, "Troubleshooting requires `graphql_call`"):
            agent.run(GraphQLAssistantGoal(goal="Something is wrong with this query", root_field="country"))

    def test_unsupported_goal_returns_planning_error(self) -> None:
        planner = FakePlanner("unsupported", "The goal is unclear.")
        sample_tool = FakeSampleTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAssistantAgent(FakeLLMClient("unused"), sample_tool, troubleshooting_tool, planner=planner)

        with self.assertRaisesRegex(AgentPlanningError, "Assistant goal must ask"):
            agent.run(
                GraphQLAssistantGoal(
                    goal="sdfdsfdf",
                    root_field="country",
                    graphql_call="query CountryQuery { country { code } }",
                )
            )

        self.assertEqual([], sample_tool.root_fields)
        self.assertEqual([], troubleshooting_tool.requests)


if __name__ == "__main__":
    unittest.main()
