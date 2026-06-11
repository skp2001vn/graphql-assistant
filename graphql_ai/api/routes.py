from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from graphql_ai.agents import AgentPlanningError, GraphQLAssistantAgent, GraphQLAssistantGoal
from graphql_ai.agents.tools import InvalidRootFieldNameError
from graphql_ai.api.schemas import (
    AssistantRequest,
    AssistantResultResponse,
    HealthResponse,
)
from graphql_ai.domain import GeneratedGraphQLSample, TroubleshootingResult


router = APIRouter()


def get_graphql_assistant_agent(request: Request) -> GraphQLAssistantAgent:
    """Return the application-scoped GraphQL AI assistant agent."""
    return request.app.state.graphql_assistant_agent


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health for local development and smoke tests."""
    return HealthResponse(status="ok")


@router.post("/assistant", response_model=AssistantResultResponse)
def run_assistant(request: Request, assistant_request: AssistantRequest) -> AssistantResultResponse:
    """Run the natural-language GraphQL AI assistant.

    The request provides a natural-language goal and a required GraphQL root
    field. The assistant uses an LLM planner to choose an assistant tool, validates
    the plan, executes the tool, and returns the final domain result.
    """
    try:
        agent = get_graphql_assistant_agent(request)
        agent_result = agent.run(
            GraphQLAssistantGoal(
                goal=assistant_request.goal,
                root_field=assistant_request.root_field,
                graphql_call=assistant_request.graphql_call,
            )
        )
    except (AgentPlanningError, InvalidRootFieldNameError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _format_assistant_result(agent_result.output)


def _format_assistant_result(output: GeneratedGraphQLSample | TroubleshootingResult) -> AssistantResultResponse:
    if isinstance(output, GeneratedGraphQLSample):
        return AssistantResultResponse(
            type="sample",
            operation=output.operation.splitlines(),
            variables=output.variables,
        )

    return AssistantResultResponse(
        type="troubleshooting",
        root_field=output.root_field,
        status=output.status,
        issues=output.issues,
        detail=output.detail,
        suggestion=output.suggestion.splitlines() if output.suggestion else [],
    )
