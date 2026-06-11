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
Choose the correct workflow intent for a user's natural-language goal.

Available intents:
- generate_sample: Generate a sample GraphQL operation for one Query or Mutation root field.
- troubleshoot: Troubleshoot a submitted GraphQL operation.

Rules:
- Return only JSON. Do not wrap the JSON in markdown.
- Use generate_sample when the user asks to generate, create, or show a sample GraphQL query or operation.
- Use troubleshoot when the user asks to fix, debug, validate, troubleshoot, or explain what is wrong with a GraphQL operation.
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


class IntentOutput(BaseModel):
    """Structured Agno output for assistant workflow selection."""

    intent: GraphQLAssistantIntent
    reason: str = Field(description="Short reason for selecting this intent.")


class AssistantPlanner(Protocol):
    """Planner interface used by the assistant agent."""

    def choose_intent(self, goal: GraphQLAssistantGoal) -> tuple[GraphQLAssistantIntent, str, str]:
        """Return the selected intent, reason, and raw planner response."""


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

    def choose_intent(self, goal: GraphQLAssistantGoal) -> tuple[GraphQLAssistantIntent, str, str]:
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

        intent, reason, raw_plan_response = self._choose_intent(normalized_goal)
        step = _build_step(normalized_goal, intent, reason)
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

    def _choose_intent(self, goal: GraphQLAssistantGoal) -> tuple[GraphQLAssistantIntent, str, str]:
        return self.planner.choose_intent(goal)


def _build_planner_input(goal: GraphQLAssistantGoal) -> str:
    return PLANNER_PROMPT_TEMPLATE.format(
        goal=goal.goal,
        root_field=goal.root_field,
        graphql_call=goal.graphql_call or "<not provided>",
    )


def _build_step(goal: GraphQLAssistantGoal, intent: GraphQLAssistantIntent, reason: str) -> AgentPlanStep:
    if intent == "generate_sample":
        tool_name = "sample_query.generate"
        inputs = {"root_field": goal.root_field}
    else:
        if goal.graphql_call is None:
            raise AgentPlanningError(
                "Troubleshooting requires `graphql_call`. Include the GraphQL operation in the request body."
            )
        tool_name = "troubleshooting.troubleshoot"
        inputs = {"root_field": goal.root_field, "graphql_call": goal.graphql_call}

    return AgentPlanStep(
        name=_tool_step_name(tool_name),
        tool_name=tool_name,
        inputs=inputs,
        reason=reason or f"Selected {intent}.",
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
