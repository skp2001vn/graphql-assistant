from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from graphql_ai.agents.tools import SampleQueryTool, TroubleshootingTool
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_ai.llm.agno_adapter import LLMClientAgnoModel
from graphql_ai.llm.base import LLMClient


GraphQLAssistantIntent = Literal["generate_sample", "troubleshoot"]
PlannerIntent = Literal["generate_sample", "troubleshoot", "unsupported"]
GraphQLAssistantOutput = GeneratedGraphQLSample | TroubleshootingResult
UNSUPPORTED_GOAL_MESSAGE = (
    "Assistant goal must ask to generate a sample GraphQL operation or troubleshoot a GraphQL operation."
)

PLANNER_SYSTEM_PROMPT = """You are a GraphQL assistant workflow planner.
Choose the correct workflow intent for a user's natural-language goal.

Available intents:
- generate_sample: Generate a sample GraphQL operation for one Query or Mutation root field.
- troubleshoot: Troubleshoot a submitted GraphQL operation.
- unsupported: The user goal is unclear, gibberish, unrelated, or not one of the supported workflows.

Rules:
- Return only JSON. Do not wrap the JSON in markdown.
- Use generate_sample when the user asks to generate, create, or show a sample GraphQL query or operation.
- Use troubleshoot when the user asks to fix, debug, validate, troubleshoot, or explain what is wrong with a GraphQL operation.
- Use unsupported when the goal is not clearly asking for sample generation or troubleshooting.
- Do not return request inputs. The application owns tool inputs.

Return a response matching the configured structured output schema.
"""

PLANNER_PROMPT_TEMPLATE = """User goal:
{goal}

Request root_field:
{root_field}

Request graphql_call:
{graphql_call}
"""


class AgentPlanningError(ValueError):
    """Raised when the assistant agent cannot build a safe executable plan."""


@dataclass(frozen=True)
class GraphQLAssistantGoal:
    """Natural-language goal submitted to the GraphQL assistant."""

    goal: str
    root_field: str
    graphql_call: str | None = None


@dataclass(frozen=True)
class GraphQLAssistantResult:
    """Result of an assistant run."""

    intent: GraphQLAssistantIntent
    goal: GraphQLAssistantGoal
    output: GraphQLAssistantOutput
    raw_plan_response: str


class IntentOutput(BaseModel):
    """Structured Agno output for assistant workflow selection."""

    intent: PlannerIntent
    reason: str = Field(description="Short reason for selecting this intent.")


class AssistantPlanner(Protocol):
    """Planner interface used by the assistant agent."""

    def choose_intent(self, goal: GraphQLAssistantGoal) -> tuple[PlannerIntent, str, str]:
        """Return the selected intent, reason, and raw planner response."""


class AgnoAssistantPlanner:
    """Agno-backed structured planner for GraphQL assistant intent selection."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Create an Agno planner over the configured application LLM client."""
        try:
            from agno.agent import Agent
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install agno with `pip install -r requirements.txt`.") from exc

        self.agent = Agent(
            model=LLMClientAgnoModel(llm_client),
            instructions=PLANNER_SYSTEM_PROMPT,
            output_schema=IntentOutput,
            parse_response=True,
            use_json_mode=True,
            telemetry=False,
        )

    def choose_intent(self, goal: GraphQLAssistantGoal) -> tuple[PlannerIntent, str, str]:
        """Run the Agno planner and return the selected assistant intent."""
        response = self.agent.run(_build_planner_input(goal))
        content = response.content
        if not isinstance(content, IntentOutput):
            raise AgentPlanningError("Planner response did not match the structured output schema.")

        raw_plan_response = content.model_dump_json()
        return content.intent, content.reason.strip(), raw_plan_response


class GraphQLAssistantAgent:
    """Application agent that plans and dispatches GraphQL assistant workflows."""

    def __init__(
        self,
        llm_client: LLMClient,
        sample_query_tool: SampleQueryTool,
        troubleshooting_tool: TroubleshootingTool,
        planner: AssistantPlanner | None = None,
    ) -> None:
        """Create an assistant agent with an LLM planner and focused tools."""
        self.sample_query_tool = sample_query_tool
        self.troubleshooting_tool = troubleshooting_tool
        self.planner = planner or AgnoAssistantPlanner(llm_client)

    def run(self, goal: GraphQLAssistantGoal) -> GraphQLAssistantResult:
        """Plan and execute the workflow needed to satisfy a GraphQL assistant goal."""
        normalized_goal = GraphQLAssistantGoal(
            goal=goal.goal.strip(),
            root_field=goal.root_field.strip(),
            graphql_call=goal.graphql_call.strip() if goal.graphql_call is not None else None,
        )
        if not normalized_goal.goal:
            raise AgentPlanningError("Assistant request `goal` must not be empty.")
        if not normalized_goal.root_field:
            raise AgentPlanningError("Assistant request `root_field` must not be empty.")

        intent, _, raw_plan_response = self.planner.choose_intent(normalized_goal)
        output = self._run_tool(intent, normalized_goal)

        return GraphQLAssistantResult(
            intent=intent,
            goal=normalized_goal,
            output=output,
            raw_plan_response=raw_plan_response,
        )

    def _run_tool(self, intent: PlannerIntent, goal: GraphQLAssistantGoal) -> GraphQLAssistantOutput:
        if intent == "generate_sample":
            return self.sample_query_tool.generate(goal.root_field)

        if intent == "unsupported":
            raise AgentPlanningError(UNSUPPORTED_GOAL_MESSAGE)

        if goal.graphql_call is None:
            raise AgentPlanningError(
                "Troubleshooting requires `graphql_call`. Include the GraphQL operation in the request body."
            )
        return self.troubleshooting_tool.troubleshoot(goal.root_field, goal.graphql_call)


def _build_planner_input(goal: GraphQLAssistantGoal) -> str:
    return PLANNER_PROMPT_TEMPLATE.format(
        goal=goal.goal,
        root_field=goal.root_field,
        graphql_call=goal.graphql_call or "<not provided>",
    )
