from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_ai.llm.base import LLMClient


GraphQLAIIntent = Literal["generate_sample", "troubleshoot"]
GraphQLAIOutput = GeneratedGraphQLSample | TroubleshootingResult

PLANNER_SYSTEM_PROMPT = """You are a GraphQL AI workflow planner.
Choose the correct tool plan for a user's natural-language goal.

Available tools:
- sample_query.generate: Generate a sample GraphQL operation for one Query or Mutation root field.
- troubleshooting.troubleshoot: Troubleshoot a submitted GraphQL operation.

Rules:
- Return only JSON. Do not wrap the JSON in markdown.
- Use exactly one tool step.
- Use sample_query.generate when the user asks to generate, create, or show a sample GraphQL query or operation.
- Use troubleshooting.troubleshoot when the user asks to fix, debug, validate, troubleshoot, or explain what is wrong with a GraphQL operation.
- The request root_field is authoritative. Do not change it.
- If the request includes graphql_call, copy it exactly into troubleshooting tool inputs.
- Do not invent a graphql_call.

Return this JSON shape:
{
  "intent": "generate_sample" | "troubleshoot",
  "steps": [
    {
      "tool_name": "sample_query.generate" | "troubleshooting.troubleshoot",
      "inputs": {
        "root_field": "request root field",
        "graphql_call": "request GraphQL call when troubleshooting"
      },
      "reason": "short reason for the tool choice"
    }
  ]
}
"""

PLANNER_PROMPT_TEMPLATE = """User goal:
{goal}

Request root_field:
{root_field}

Request graphql_call:
{graphql_call}
"""

CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", flags=re.DOTALL)


class AgentPlanningError(ValueError):
    """Raised when the application agent cannot build a safe executable plan."""


class SampleQueryTool(Protocol):
    """Tool interface for generating sample GraphQL operations."""

    def generate(self, root_field: str) -> GeneratedGraphQLSample:
        """Generate a sample GraphQL operation for a root field."""


class TroubleshootingTool(Protocol):
    """Tool interface for troubleshooting submitted GraphQL operations."""

    def troubleshoot(self, root_field: str, graphql_call: str) -> TroubleshootingResult:
        """Troubleshoot a submitted GraphQL operation."""


@dataclass(frozen=True)
class GraphQLAIGoal:
    """Natural-language goal submitted to the application-level GraphQL AI agent."""

    goal: str
    root_field: str
    graphql_call: str | None = None


@dataclass(frozen=True)
class AgentPlanStep:
    """Single LLM-planned step the agent will execute with a service tool."""

    name: str
    tool_name: str
    inputs: dict[str, str]
    reason: str


@dataclass(frozen=True)
class ToolCall:
    """Structured record of a tool invocation made by the agent."""

    tool_name: str
    inputs: dict[str, str]


@dataclass(frozen=True)
class ToolObservation:
    """Structured tool output summary passed back into the agent state."""

    tool_name: str
    output_type: str
    summary: str


@dataclass(frozen=True)
class GraphQLAIResult:
    """Result of an application-agent run, including plan and tool observations."""

    intent: GraphQLAIIntent
    goal: GraphQLAIGoal
    plan: tuple[AgentPlanStep, ...]
    tool_calls: tuple[ToolCall, ...]
    observations: tuple[ToolObservation, ...]
    output: GraphQLAIOutput
    raw_plan_response: str


class GraphQLAIAgent:
    """Smart application agent that plans and dispatches GraphQL AI workflows.

    The agent uses LLM inference to convert a natural-language goal into an
    executable tool plan. It validates the planner output before calling any
    service tool, records tool calls and tool observations, and returns the
    selected domain output. Services still own business logic such as RAG,
    prompt construction, inference, and guardrails; this agent owns natural
    language planning, tool selection, and orchestration.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        sample_query_tool: SampleQueryTool,
        troubleshooting_tool: TroubleshootingTool,
    ) -> None:
        """Create an application agent with an LLM planner and focused service tools."""
        self.llm_client = llm_client
        self.sample_query_tool = sample_query_tool
        self.troubleshooting_tool = troubleshooting_tool

    def run(self, goal: GraphQLAIGoal) -> GraphQLAIResult:
        """Plan and execute the workflow needed to satisfy a GraphQL AI goal."""
        normalized_goal = self._validate_goal(goal)
        raw_plan_response = self.llm_client.generate(self._build_planner_prompt(normalized_goal))
        intent, plan = self._parse_and_validate_plan(normalized_goal, raw_plan_response)
        tool_calls: list[ToolCall] = []
        observations: list[ToolObservation] = []
        output: GraphQLAIOutput | None = None

        for step in plan:
            tool_call = ToolCall(tool_name=step.tool_name, inputs=dict(step.inputs))
            tool_calls.append(tool_call)
            output = self._execute_tool(tool_call)
            observations.append(self._observe(tool_call.tool_name, output))

        if output is None:
            raise AgentPlanningError(f"Agent plan did not produce output for intent: {intent}")

        return GraphQLAIResult(
            intent=intent,
            goal=normalized_goal,
            plan=plan,
            tool_calls=tuple(tool_calls),
            observations=tuple(observations),
            output=output,
            raw_plan_response=raw_plan_response,
        )

    def _validate_goal(self, goal: GraphQLAIGoal) -> GraphQLAIGoal:
        normalized_text = goal.goal.strip()
        normalized_root_field = goal.root_field.strip()
        normalized_graphql_call = goal.graphql_call.strip() if goal.graphql_call is not None else None

        if not normalized_text:
            raise AgentPlanningError("Assistant request `goal` must not be empty.")
        if not normalized_root_field:
            raise AgentPlanningError("Assistant request `root_field` must not be empty.")

        return GraphQLAIGoal(
            goal=normalized_text,
            root_field=normalized_root_field,
            graphql_call=normalized_graphql_call or None,
        )

    def _build_planner_prompt(self, goal: GraphQLAIGoal) -> str:
        user_prompt = PLANNER_PROMPT_TEMPLATE.format(
            goal=goal.goal,
            root_field=goal.root_field,
            graphql_call=goal.graphql_call or "<not provided>",
        )
        return f"{PLANNER_SYSTEM_PROMPT}\n\n{user_prompt}"

    def _parse_and_validate_plan(
        self,
        goal: GraphQLAIGoal,
        raw_response: str,
    ) -> tuple[GraphQLAIIntent, tuple[AgentPlanStep, ...]]:
        payload = _parse_planner_json(raw_response)
        intent = payload.get("intent")
        if intent not in {"generate_sample", "troubleshoot"}:
            raise AgentPlanningError("Planner returned unsupported `intent`.")

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != 1:
            raise AgentPlanningError("Planner must return exactly one tool step.")

        step = self._parse_plan_step(raw_steps[0])
        self._validate_step_for_goal(goal, intent, step)
        return intent, (step,)

    def _parse_plan_step(self, raw_step: object) -> AgentPlanStep:
        if not isinstance(raw_step, dict):
            raise AgentPlanningError("Planner step must be an object.")

        tool_name = raw_step.get("tool_name")
        if tool_name not in {"sample_query.generate", "troubleshooting.troubleshoot"}:
            raise AgentPlanningError("Planner returned an unsupported tool.")

        raw_inputs = raw_step.get("inputs")
        if not isinstance(raw_inputs, dict):
            raise AgentPlanningError("Planner step must include an `inputs` object.")

        inputs = {}
        for key, value in raw_inputs.items():
            if isinstance(key, str) and isinstance(value, str):
                inputs[key] = value

        reason = raw_step.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise AgentPlanningError("Planner step must include a non-empty `reason`.")

        return AgentPlanStep(
            name=_tool_step_name(tool_name),
            tool_name=tool_name,
            inputs=inputs,
            reason=reason.strip(),
        )

    def _validate_step_for_goal(
        self,
        goal: GraphQLAIGoal,
        intent: GraphQLAIIntent,
        step: AgentPlanStep,
    ) -> None:
        expected_tool = {
            "generate_sample": "sample_query.generate",
            "troubleshoot": "troubleshooting.troubleshoot",
        }[intent]
        if step.tool_name != expected_tool:
            raise AgentPlanningError("Planner intent and tool do not match.")

        planned_root_field = step.inputs.get("root_field")
        if planned_root_field != goal.root_field:
            raise AgentPlanningError(
                f"Planner must use request `root_field` exactly: {goal.root_field}."
            )

        if intent == "generate_sample":
            return

        if goal.graphql_call is None:
            raise AgentPlanningError(
                "Troubleshooting requires `graphql_call`. Include the GraphQL operation in the request body."
            )

        planned_graphql_call = step.inputs.get("graphql_call")
        if planned_graphql_call != goal.graphql_call:
            raise AgentPlanningError("Planner must use request `graphql_call` exactly.")

    def _execute_tool(self, tool_call: ToolCall) -> GraphQLAIOutput:
        if tool_call.tool_name == "sample_query.generate":
            return self.sample_query_tool.generate(tool_call.inputs["root_field"])

        if tool_call.tool_name == "troubleshooting.troubleshoot":
            return self.troubleshooting_tool.troubleshoot(
                tool_call.inputs["root_field"],
                tool_call.inputs["graphql_call"],
            )

        raise AgentPlanningError(f"Unsupported agent tool: {tool_call.tool_name}")

    def _observe(self, tool_name: str, output: GraphQLAIOutput) -> ToolObservation:
        if isinstance(output, GeneratedGraphQLSample):
            return ToolObservation(
                tool_name=tool_name,
                output_type="GeneratedGraphQLSample",
                summary=(
                    f"Generated operation with {len(output.operation.splitlines())} lines "
                    f"and {len(output.variables)} variables."
                ),
            )

        return ToolObservation(
            tool_name=tool_name,
            output_type="TroubleshootingResult",
            summary=(
                f"Troubleshooting finished with status {output.status}, "
                f"{len(output.issues)} issues, and "
                f"{'a suggestion' if output.suggestion else 'no suggestion'}."
            ),
        )


def _parse_planner_json(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    code_blocks = CODE_BLOCK_RE.findall(text)
    if code_blocks:
        text = code_blocks[0].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentPlanningError("Planner response must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise AgentPlanningError("Planner response JSON must be an object.")

    return payload


def _tool_step_name(tool_name: str) -> str:
    if tool_name == "sample_query.generate":
        return "Generate sample GraphQL operation"
    if tool_name == "troubleshooting.troubleshoot":
        return "Troubleshoot submitted GraphQL operation"
    return tool_name
