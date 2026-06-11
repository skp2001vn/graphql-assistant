from __future__ import annotations

import unittest

from graphql_ai.agents import AgentPlanningError, GraphQLAssistantAgent, GraphQLAssistantGoal
from graphql_ai.agents.assistant_agent import AgentPlanStep, AgnoAssistantPlanner
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeSampleQueryTool:
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
    def __init__(self, intent: str, step: AgentPlanStep, raw_response: str = "{}") -> None:
        self.intent = intent
        self.step = step
        self.raw_response = raw_response
        self.goals: list[GraphQLAssistantGoal] = []

    def plan(self, goal: GraphQLAssistantGoal) -> tuple[str, AgentPlanStep, str]:
        self.goals.append(goal)
        return self.intent, self.step, self.raw_response


class GraphQLAssistantAgentTest(unittest.TestCase):
    def test_agno_planner_uses_structured_output(self) -> None:
        llm_client = FakeLLMClient(
            '{"intent":"generate_sample","steps":[{"tool_name":"sample_query.generate",'
            '"inputs":{"root_field":"country"},"reason":"The user asked for a sample query."}]}'
        )

        intent, step, raw_response = AgnoAssistantPlanner(llm_client).plan(
            GraphQLAssistantGoal(goal="Generate a sample query", root_field="country")
        )

        self.assertIn("Generate a sample query", llm_client.prompts[0])
        self.assertEqual("generate_sample", intent)
        self.assertEqual("sample_query.generate", step.tool_name)
        self.assertEqual({"root_field": "country"}, step.inputs)
        self.assertIn("generate_sample", raw_response)

    def test_sample_goal_uses_plan_and_calls_sample_tool(self) -> None:
        planner = FakePlanner(
            "generate_sample",
            AgentPlanStep(
                name="Generate sample GraphQL operation",
                tool_name="sample_query.generate",
                inputs={"root_field": "country"},
                reason="The user asked for a sample query.",
            ),
        )
        sample_tool = FakeSampleQueryTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAssistantAgent(FakeLLMClient("unused"), sample_tool, troubleshooting_tool, planner=planner)

        result = agent.run(GraphQLAssistantGoal(goal="Generate a sample query", root_field="country"))

        self.assertEqual([GraphQLAssistantGoal(goal="Generate a sample query", root_field="country")], planner.goals)
        self.assertEqual(["country"], sample_tool.root_fields)
        self.assertEqual([], troubleshooting_tool.requests)
        self.assertEqual("generate_sample", result.intent)
        self.assertEqual("sample_query.generate", result.plan[0].tool_name)
        self.assertEqual("sample_query.generate", result.tool_calls[0].tool_name)
        self.assertIsInstance(result.output, GeneratedGraphQLSample)

    def test_troubleshoot_goal_uses_plan_and_calls_troubleshooting_tool(self) -> None:
        graphql_call = "query CountyQuery($code: ID!) { county(code: $code) { code } }"
        planner = FakePlanner(
            "troubleshoot",
            AgentPlanStep(
                name="Troubleshoot submitted GraphQL operation",
                tool_name="troubleshooting.troubleshoot",
                inputs={"root_field": "country", "graphql_call": graphql_call},
                reason="The user asked to fix a GraphQL operation.",
            ),
        )
        sample_tool = FakeSampleQueryTool()
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
        self.assertEqual("troubleshooting.troubleshoot", result.plan[0].tool_name)
        self.assertIsInstance(result.output, TroubleshootingResult)

    def test_troubleshoot_plan_requires_request_graphql_call(self) -> None:
        planner = FakePlanner(
            "troubleshoot",
            AgentPlanStep(
                name="Troubleshoot submitted GraphQL operation",
                tool_name="troubleshooting.troubleshoot",
                inputs={"root_field": "country"},
                reason="The user asked to troubleshoot.",
            ),
        )
        agent = GraphQLAssistantAgent(
            FakeLLMClient("unused"),
            FakeSampleQueryTool(),
            FakeTroubleshootingTool(),
            planner=planner,
        )

        with self.assertRaisesRegex(AgentPlanningError, "Troubleshooting requires `graphql_call`"):
            agent.run(GraphQLAssistantGoal(goal="Something is wrong with this query", root_field="country"))

    def test_rejects_planner_root_field_change(self) -> None:
        planner = FakePlanner(
            "generate_sample",
            AgentPlanStep(
                name="Generate sample GraphQL operation",
                tool_name="sample_query.generate",
                inputs={"root_field": "countries"},
                reason="The user asked for a sample query.",
            ),
        )
        agent = GraphQLAssistantAgent(
            FakeLLMClient("unused"),
            FakeSampleQueryTool(),
            FakeTroubleshootingTool(),
            planner=planner,
        )

        with self.assertRaisesRegex(AgentPlanningError, "root_field"):
            agent.run(GraphQLAssistantGoal(goal="Generate a sample query", root_field="country"))


if __name__ == "__main__":
    unittest.main()
