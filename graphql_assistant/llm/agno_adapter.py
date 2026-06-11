from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from graphql_assistant.llm.base import LLMClient


class LLMClientAgnoModel:
    """Adapter from the app's `LLMClient` protocol to Agno's model interface.

    The rest of the application talks to providers through the small local
    `LLMClient` protocol, while Agno expects an object that implements its own
    `Model` contract. This adapter bridges those two shapes so the assistant
    layer can use Agno for planning without rewriting the provider stack.

    The adapter is intentionally narrow:

    1. flatten Agno messages into one prompt string
    2. send that prompt through the existing `LLMClient`
    3. wrap the returned text in Agno's `ModelResponse`

    It does not currently implement provider-native tool calling, structured
    response transport, or richer multimodal semantics. In this codebase that
    is intentional: Agno is presently used as a structured orchestration layer
    over a plain text-generation boundary, not as a fully provider-native agent
    runtime.
    """

    def __new__(cls, llm_client: LLMClient) -> Any:
        try:
            from agno.models.base import Model
            from agno.models.response import ModelResponse
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install agno with `pip install -r requirements.txt`.") from exc

        class _LLMClientAgnoModel(Model):
            def __init__(self, wrapped_llm_client: LLMClient) -> None:
                super().__init__(id="graphql-assistant-llm-client", provider="graphql_assistant")
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


def _format_agno_messages(messages: list[Any]) -> str:
    """Flatten Agno message objects into the prompt format used by `LLMClient`.

    The local provider clients accept one prompt string rather than a chat
    message array. This helper preserves the role ordering from Agno's message
    list and converts each message into a readable text block so the wrapped
    provider still receives the full conversation context in sequence.
    """
    prompt_parts = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        if content:
            prompt_parts.append(f"{getattr(message, 'role', 'message').upper()}:\n{content}")

    return "\n\n".join(prompt_parts)
