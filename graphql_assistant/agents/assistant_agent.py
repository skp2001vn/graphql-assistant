from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from graphql_assistant.agents.tools import SampleTool, TroubleshootingTool
from graphql_assistant.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_assistant.llm.agno_adapter import LLMClientAgnoModel
from graphql_assistant.llm.base import LLMClient


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
    """Structured planner payload for assistant intent classification.

    The assistant uses Agno in structured-output mode rather than letting the
    framework execute tools directly. This schema limits the model's role to
    intent classification, which keeps routing auditable and easy to test.
    """

    intent: PlannerIntent
    reason: str = Field(description="Short reason for selecting this intent.")


class AgnoAssistantPlanner:
    """Agno-backed planner that classifies the user's requested workflow.

    This component is intentionally narrow. It uses Agno as a structured
    orchestration layer for intent selection, not as a general autonomous
    agent. The planner reads the natural-language goal plus request context and
    maps it to one of three intents:

    - `generate_sample`
    - `troubleshoot`
    - `unsupported`

    The application still owns tool inputs and execution. That split keeps the
    LLM focused on semantic classification while deterministic Python code
    enforces request contracts and dispatch behavior.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Create a structured Agno planner over the configured LLM client.

        The planner runs in JSON/typed-output mode so the model must emit a
        payload matching `IntentOutput`. This is the main technique used to
        reduce brittle string parsing in the assistant layer.
        """
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
        """Classify the request into an executable assistant workflow intent.

        The returned raw payload is preserved for observability and debugging,
        while the typed intent is used for deterministic tool dispatch.
        """
        response = self.agent.run(_build_planner_input(goal))
        content = response.content
        if not isinstance(content, IntentOutput):
            raise AgentPlanningError("Planner response did not match the structured output schema.")

        raw_plan_response = content.model_dump_json()
        return content.intent, content.reason.strip(), raw_plan_response


class GraphQLAssistantAgent:
    """Top-level application agent for GraphQL assistant workflows.

    This class is the business entry point behind the unified `/assistant`
    interface. It implements a small planner-dispatcher pattern:

    1. Normalize and validate the user request envelope.
    2. Ask an Agno-backed planner to classify the user's goal.
    3. Dispatch to the matching assistant tool.
    4. Return a typed application result that includes the selected intent and
       the tool output.

    The assistant deliberately does not expose direct framework-managed tool
    execution. Agno is used only for structured workflow planning. All request
    validation, tool input ownership, and business guardrails remain in
    application code so the behavior stays explicit, testable, and stable as
    more assistant workflows are added.
    """

    def __init__(
        self,
        sample_tool: SampleTool,
        troubleshooting_tool: TroubleshootingTool,
        planner: AgnoAssistantPlanner,
    ) -> None:
        """Create the assistant with a planner and concrete workflow tools.

        The assistant depends on specialized tools rather than a generic tool
        registry because the current business scope is small and explicit:
        sample generation and troubleshooting.
        """
        self.sample_tool = sample_tool
        self.troubleshooting_tool = troubleshooting_tool
        self.planner = planner

    def run(self, goal: GraphQLAssistantGoal) -> GraphQLAssistantResult:
        """Plan and execute the workflow for a single assistant request.

        This method is the orchestration boundary for the app's assistant use
        cases. It performs lightweight request normalization, asks the planner
        to classify the natural-language goal, and then executes the matching
        tool with application-owned inputs.

        The separation matters operationally:
        - the planner handles semantic intent detection,
        - tools handle domain workflows such as RAG retrieval, inference, and
          GraphQL validation,
        - and the assistant preserves a single API contract for both.
        """
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
            return self.sample_tool.generate(goal.root_field)

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
