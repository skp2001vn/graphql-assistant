from __future__ import annotations

import unittest

from graphql_ai.agents import AgentPlanningError, GraphQLAIAgent, GraphQLAIGoal
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


class GraphQLAIAgentTest(unittest.TestCase):
    def test_sample_goal_uses_llm_plan_and_calls_sample_tool(self) -> None:
        llm_client = FakeLLMClient(
            """
{
  "intent": "generate_sample",
  "steps": [
    {
      "tool_name": "sample_query.generate",
      "inputs": {"root_field": "country"},
      "reason": "The user asked for a sample query."
    }
  ]
}
"""
        )
        sample_tool = FakeSampleQueryTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAIAgent(llm_client, sample_tool, troubleshooting_tool)

        result = agent.run(GraphQLAIGoal(goal="Generate a sample query", root_field="country"))

        self.assertIn("Generate a sample query", llm_client.prompts[0])
        self.assertEqual(["country"], sample_tool.root_fields)
        self.assertEqual([], troubleshooting_tool.requests)
        self.assertEqual("generate_sample", result.intent)
        self.assertEqual("sample_query.generate", result.plan[0].tool_name)
        self.assertEqual("sample_query.generate", result.tool_calls[0].tool_name)
        self.assertEqual("GeneratedGraphQLSample", result.observations[0].output_type)
        self.assertIsInstance(result.output, GeneratedGraphQLSample)

    def test_troubleshoot_goal_uses_llm_plan_and_calls_troubleshooting_tool(self) -> None:
        graphql_call = "query CountyQuery($code: ID!) { county(code: $code) { code } }"
        llm_client = FakeLLMClient(
            f"""
{{
  "intent": "troubleshoot",
  "steps": [
    {{
      "tool_name": "troubleshooting.troubleshoot",
      "inputs": {{"root_field": "country", "graphql_call": "{graphql_call}"}},
      "reason": "The user asked to fix a GraphQL operation."
    }}
  ]
}}
"""
        )
        sample_tool = FakeSampleQueryTool()
        troubleshooting_tool = FakeTroubleshootingTool()
        agent = GraphQLAIAgent(llm_client, sample_tool, troubleshooting_tool)

        result = agent.run(
            GraphQLAIGoal(
                goal="Something is wrong with this query",
                root_field="country",
                graphql_call=graphql_call,
            )
        )

        self.assertEqual([], sample_tool.root_fields)
        self.assertEqual([("country", graphql_call)], troubleshooting_tool.requests)
        self.assertEqual("troubleshoot", result.intent)
        self.assertEqual("troubleshooting.troubleshoot", result.plan[0].tool_name)
        self.assertEqual("TroubleshootingResult", result.observations[0].output_type)
        self.assertIsInstance(result.output, TroubleshootingResult)

    def test_troubleshoot_plan_requires_request_graphql_call(self) -> None:
        llm_client = FakeLLMClient(
            """
{
  "intent": "troubleshoot",
  "steps": [
    {
      "tool_name": "troubleshooting.troubleshoot",
      "inputs": {"root_field": "country"},
      "reason": "The user asked to troubleshoot."
    }
  ]
}
"""
        )
        agent = GraphQLAIAgent(llm_client, FakeSampleQueryTool(), FakeTroubleshootingTool())

        with self.assertRaisesRegex(AgentPlanningError, "Troubleshooting requires `graphql_call`"):
            agent.run(GraphQLAIGoal(goal="Something is wrong with this query", root_field="country"))

    def test_rejects_planner_root_field_change(self) -> None:
        llm_client = FakeLLMClient(
            """
{
  "intent": "generate_sample",
  "steps": [
    {
      "tool_name": "sample_query.generate",
      "inputs": {"root_field": "countries"},
      "reason": "The user asked for a sample query."
    }
  ]
}
"""
        )
        agent = GraphQLAIAgent(llm_client, FakeSampleQueryTool(), FakeTroubleshootingTool())

        with self.assertRaisesRegex(AgentPlanningError, "root_field"):
            agent.run(GraphQLAIGoal(goal="Generate a sample query", root_field="country"))


if __name__ == "__main__":
    unittest.main()
