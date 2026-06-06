from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    """Natural-language request for the GraphQL AI assistant."""

    goal: str = Field(min_length=1, description="Natural-language assistant goal.")
    root_field: str = Field(
        min_length=1,
        description="GraphQL Query or Mutation root field the request focuses on.",
    )
    graphql_call: str | None = Field(
        default=None,
        description="GraphQL operation to troubleshoot when the assistant chooses troubleshooting.",
    )


class AgentPlanStepResponse(BaseModel):
    """HTTP representation of one assistant plan step."""

    name: str
    tool_name: str
    reason: str


class ToolCallResponse(BaseModel):
    """HTTP representation of one assistant tool call."""

    tool_name: str
    inputs: dict[str, str]


class ToolObservationResponse(BaseModel):
    """HTTP representation of one assistant tool observation."""

    tool_name: str
    output_type: str
    summary: str


class AssistantResultResponse(BaseModel):
    """HTTP response for the GraphQL AI assistant."""

    intent: str
    plan: list[AgentPlanStepResponse]
    tool_calls: list[ToolCallResponse]
    observations: list[ToolObservationResponse]
    result: dict[str, Any]


class HealthResponse(BaseModel):
    """HTTP response for the health endpoint."""

    status: Literal["ok"]
