"""OpenAI Responses API adapter.

Supports the newer /v1/responses endpoint (different from Chat Completions).
API docs: https://platform.openai.com/docs/api-reference/responses
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llm_adapters.config import ProviderConfig
from llm_adapters.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from llm_adapters.models import ChatRequest, ChatResponse, Role, StreamDelta, Usage
from llm_adapters.providers.base import BaseLLMProvider


class OpenAIResponsesProvider(BaseLLMProvider):
    """OpenAI Responses API adapter (/v1/responses)."""

    name = "openai-responses"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _build_input(self, request: ChatRequest) -> list[dict]:
        """Convert unified messages to Responses API input format."""
        items: list[dict] = []
        for m in request.messages:
            if m.role == Role.SYSTEM:
                # System messages become developer role in Responses API
                items.append({
                    "role": "developer",
                    "content": m.content,
                })
            else:
                items.append({
                    "role": m.role.value,
                    "content": m.content,
                })
        return items

    def _build_payload(self, request: ChatRequest, stream: bool = False) -> dict:
        payload: dict = {
            "model": request.model,
            "input": self._build_input(request),
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        return payload

    def _extract_text(self, data: dict) -> str:
        """Extract text from Responses API output array."""
        output = data.get("output", [])
        parts: list[str] = []
        for item in output:
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        parts.append(content.get("text", ""))
        return "".join(parts)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)
        resp = await self._client.post("/v1/responses", json=payload)
        self._check_error(resp)
        data = resp.json()
        usage = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=self._extract_text(data),
            finish_reason=data.get("status", ""),
            usage=Usage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamDelta]:
        payload = self._build_payload(request, stream=True)
        async with self._client.stream("POST", "/v1/responses", json=payload) as resp:
            self._check_error(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                data = json.loads(data_str)
                event_type = data.get("type", "")

                if event_type == "response.output_text.delta":
                    yield StreamDelta(content=data.get("delta", ""))

                elif event_type == "response.completed":
                    resp_data = data.get("response", {})
                    usage = resp_data.get("usage", {})
                    yield StreamDelta(
                        finish_reason=resp_data.get("status", "completed"),
                        usage=Usage(
                            prompt_tokens=usage.get("input_tokens", 0),
                            completion_tokens=usage.get("output_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        ),
                    )

    def _check_error(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        if resp.status_code == 401:
            raise AuthenticationError(
                "Invalid OpenAI API key", provider=self.name, status_code=401
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "OpenAI rate limit exceeded", provider=self.name, status_code=429
            )
        raise ProviderError(
            f"OpenAI Responses API error {resp.status_code}: {resp.text}",
            provider=self.name,
            status_code=resp.status_code,
        )

    async def close(self) -> None:
        await self._client.aclose()
