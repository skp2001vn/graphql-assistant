from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from graphql_ai.agents import AgentPlanningError, GraphQLAIAgent, GraphQLAIGoal
from graphql_ai.api.schemas import (
    AgentPlanStepResponse,
    AssistantRequest,
    AssistantResultResponse,
    HealthResponse,
    ToolCallResponse,
    ToolObservationResponse,
)
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult
from graphql_ai.services.sample_query_service import InvalidRootFieldNameError


router = APIRouter()


def get_graphql_ai_agent(request: Request) -> GraphQLAIAgent:
    """Return the application-scoped GraphQL AI assistant agent."""
    return request.app.state.graphql_ai_agent


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health for local development and smoke tests."""
    return HealthResponse(status="ok")


@router.post("/assistant", response_model=AssistantResultResponse)
def run_assistant(request: Request, assistant_request: AssistantRequest) -> AssistantResultResponse:
    """Run the natural-language GraphQL AI assistant.

    The request provides a natural-language goal and a required GraphQL root
    field. The assistant uses an LLM planner to choose a service tool, validates
    the plan, executes the tool, and returns the plan, tool calls, observations,
    and final domain result.
    """
    try:
        agent = get_graphql_ai_agent(request)
        agent_result = agent.run(
            GraphQLAIGoal(
                goal=assistant_request.goal,
                root_field=assistant_request.root_field,
                graphql_call=assistant_request.graphql_call,
            )
        )
    except (AgentPlanningError, InvalidRootFieldNameError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AssistantResultResponse(
        intent=agent_result.intent,
        plan=[
            AgentPlanStepResponse(
                name=step.name,
                tool_name=step.tool_name,
                reason=step.reason,
            )
            for step in agent_result.plan
        ],
        tool_calls=[
            ToolCallResponse(
                tool_name=tool_call.tool_name,
                inputs=tool_call.inputs,
            )
            for tool_call in agent_result.tool_calls
        ],
        observations=[
            ToolObservationResponse(
                tool_name=observation.tool_name,
                output_type=observation.output_type,
                summary=observation.summary,
            )
            for observation in agent_result.observations
        ],
        result=_format_assistant_result(agent_result.output),
    )


def _format_assistant_result(output: GeneratedGraphQLSample | TroubleshootingResult) -> dict[str, object]:
    if isinstance(output, GeneratedGraphQLSample):
        return {
            "type": "sample",
            "operation": output.operation.splitlines(),
            "variables": output.variables,
        }

    return {
        "type": "troubleshooting",
        "root_field": output.root_field,
        "status": output.status,
        "issues": output.issues,
        "detail": output.detail,
        "suggestion": output.suggestion.splitlines() if output.suggestion else [],
    }
