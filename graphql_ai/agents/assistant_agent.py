from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from graphql_ai.agents.tools import SampleQueryTool, TroubleshootingTool
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_ai.llm.base import LLMClient


GraphQLAssistantIntent = Literal["generate_sample", "troubleshoot"]
GraphQLAssistantOutput = GeneratedGraphQLSample | TroubleshootingResult

PLANNER_SYSTEM_PROMPT = """You are a GraphQL assistant workflow planner.
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
class AgentPlanStep:
    """Single LLM-planned step the assistant will execute with a tool."""

    name: str
    tool_name: str
    inputs: dict[str, str]
    reason: str


@dataclass(frozen=True)
class ToolCall:
    """Structured record of a tool invocation made by the assistant."""

    tool_name: str
    inputs: dict[str, str]


@dataclass(frozen=True)
class GraphQLAssistantResult:
    """Result of an assistant run, including the selected plan and output."""

    intent: GraphQLAssistantIntent
    goal: GraphQLAssistantGoal
    plan: tuple[AgentPlanStep, ...]
    tool_calls: tuple[ToolCall, ...]
    output: GraphQLAssistantOutput
    raw_plan_response: str


class PlannerStepOutput(BaseModel):
    """Structured Agno planner output for one assistant tool step."""

    tool_name: Literal["sample_query.generate", "troubleshooting.troubleshoot"] = Field(
        description="Assistant tool selected for the request."
    )
    inputs: dict[str, str] = Field(description="Tool inputs copied from the request.")
    reason: str = Field(description="Short reason for selecting this tool.")


class PlannerOutput(BaseModel):
    """Structured Agno planner output for the assistant workflow."""

    intent: GraphQLAssistantIntent
    steps: list[PlannerStepOutput]


class AssistantPlanner(Protocol):
    """Planner interface used by the assistant agent."""

    def plan(self, goal: GraphQLAssistantGoal) -> tuple[GraphQLAssistantIntent, AgentPlanStep, str]:
        """Return a validated intent, step, and raw planner response."""


class LLMClientAgnoModel:
    """Lazy Agno model wrapper around the app's existing LLM client."""

    def __new__(cls, llm_client: LLMClient) -> Any:
        try:
            from agno.models.base import Model
            from agno.models.response import ModelResponse
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install agno with `pip install -r requirements.txt`.") from exc

        class _LLMClientAgnoModel(Model):
            def __init__(self, wrapped_llm_client: LLMClient) -> None:
                super().__init__(id="graphql-ai-llm-client", provider="graphql_ai")
                self.wrapped_llm_client = wrapped_llm_client

            def response(
                self,
                messages: list[Any],
                response_format: dict[str, Any] | type[BaseModel] | None = None,
                tools: list[Any] | None = None,
                tool_choice: str | dict[str, Any] | None = None,
                tool_call_limit: int | None = None,
                run_response: Any | None = None,
                send_media_to_model: bool = True,
                compression_manager: Any | None = None,
            ) -> Any:
                prompt = _format_agno_messages(messages)
                return ModelResponse(role="assistant", content=self.wrapped_llm_client.generate(prompt))

            def invoke(self, *args: Any, **kwargs: Any) -> Any:
                return self.response(*args, **kwargs)

            async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
                return self.invoke(*args, **kwargs)

            def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
                yield self.invoke(*args, **kwargs)

            async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
                yield self.invoke(*args, **kwargs)

            def _parse_provider_response(self, response: Any, **kwargs: Any) -> Any:
                return response

            def _parse_provider_response_delta(self, response_delta: Any) -> Any:
                return response_delta

        return _LLMClientAgnoModel(llm_client)


class AgnoAssistantPlanner:
    """Agno-backed structured planner for GraphQL assistant tool selection."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Create an Agno planner over the configured application LLM client."""
        try:
            from agno.agent import Agent
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install agno with `pip install -r requirements.txt`.") from exc

        self.agent = Agent(
            model=LLMClientAgnoModel(llm_client),
            instructions=PLANNER_SYSTEM_PROMPT,
            output_schema=PlannerOutput,
            parse_response=True,
            use_json_mode=True,
            telemetry=False,
        )

    def plan(self, goal: GraphQLAssistantGoal) -> tuple[GraphQLAssistantIntent, AgentPlanStep, str]:
        """Run the Agno planner and return the selected assistant tool step."""
        response = self.agent.run(_build_planner_input(goal))
        content = response.content
        if not isinstance(content, PlannerOutput):
            raise AgentPlanningError("Planner response did not match the structured output schema.")

        raw_plan_response = content.model_dump_json()
        if len(content.steps) != 1:
            raise AgentPlanningError("Planner must return exactly one tool step.")

        step_output = content.steps[0]
        if not step_output.reason.strip():
            raise AgentPlanningError("Planner step must include a non-empty `reason`.")

        step = AgentPlanStep(
            name=_tool_step_name(step_output.tool_name),
            tool_name=step_output.tool_name,
            inputs=dict(step_output.inputs),
            reason=step_output.reason.strip(),
        )
        return content.intent, step, raw_plan_response


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

        intent, step, raw_plan_response = self.planner.plan(normalized_goal)
        self._validate_planned_step(normalized_goal, intent, step)
        tool_call = ToolCall(tool_name=step.tool_name, inputs=dict(step.inputs))

        if step.tool_name == "sample_query.generate":
            output: GraphQLAssistantOutput = self.sample_query_tool.generate(step.inputs["root_field"])
        elif step.tool_name == "troubleshooting.troubleshoot":
            output = self.troubleshooting_tool.troubleshoot(
                step.inputs["root_field"],
                step.inputs["graphql_call"],
            )
        else:
            raise AgentPlanningError(f"Unsupported assistant tool: {step.tool_name}")

        return GraphQLAssistantResult(
            intent=intent,
            goal=normalized_goal,
            plan=(step,),
            tool_calls=(tool_call,),
            output=output,
            raw_plan_response=raw_plan_response,
        )

    def _validate_planned_step(
        self,
        goal: GraphQLAssistantGoal,
        intent: GraphQLAssistantIntent,
        step: AgentPlanStep,
    ) -> None:
        expected_tool = {
            "generate_sample": "sample_query.generate",
            "troubleshoot": "troubleshooting.troubleshoot",
        }[intent]
        if step.tool_name != expected_tool:
            raise AgentPlanningError("Planner intent and tool do not match.")
        if step.inputs.get("root_field") != goal.root_field:
            raise AgentPlanningError(f"Planner must use request `root_field` exactly: {goal.root_field}.")
        if intent == "troubleshoot":
            if goal.graphql_call is None:
                raise AgentPlanningError(
                    "Troubleshooting requires `graphql_call`. Include the GraphQL operation in the request body."
                )
            if step.inputs.get("graphql_call") != goal.graphql_call:
                raise AgentPlanningError("Planner must use request `graphql_call` exactly.")


def _build_planner_input(goal: GraphQLAssistantGoal) -> str:
    return PLANNER_PROMPT_TEMPLATE.format(
        goal=goal.goal,
        root_field=goal.root_field,
        graphql_call=goal.graphql_call or "<not provided>",
    )


def _format_agno_messages(messages: list[Any]) -> str:
    prompt_parts = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        if content:
            prompt_parts.append(f"{getattr(message, 'role', 'message').upper()}:\n{content}")

    return "\n\n".join(prompt_parts)


def _tool_step_name(tool_name: str) -> str:
    if tool_name == "sample_query.generate":
        return "Generate sample GraphQL operation"
    if tool_name == "troubleshooting.troubleshoot":
        return "Troubleshoot submitted GraphQL operation"
    return tool_name
