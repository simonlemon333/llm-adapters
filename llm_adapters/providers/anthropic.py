"""Anthropic Messages API adapter.

Supports: Claude 4, Claude 3.5, etc.
API docs: https://docs.anthropic.com/en/api/messages
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

_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Messages API adapter."""

    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            },
        )

    def _build_payload(self, request: ChatRequest, stream: bool = False) -> dict:
        # Anthropic separates system message from the messages array
        system_text = ""
        messages = []
        for m in request.messages:
            if m.role == Role.SYSTEM:
                system_text = m.content
            else:
                messages.append({"role": m.role.value, "content": m.content})

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": stream,
        }
        if system_text:
            payload["system"] = system_text
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)
        resp = await self._client.post("/v1/messages", json=payload)
        self._check_error(resp)
        data = resp.json()
        content_blocks = data.get("content", [])
        text = "".join(b["text"] for b in content_blocks if b["type"] == "text")
        usage = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=text,
            finish_reason=data.get("stop_reason"),
            usage=Usage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamDelta]:
        payload = self._build_payload(request, stream=True)
        async with self._client.stream("POST", "/v1/messages", json=payload) as resp:
            self._check_error(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                event_type = data.get("type", "")

                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamDelta(content=delta.get("text", ""))

                elif event_type == "message_delta":
                    delta = data.get("delta", {})
                    usage = data.get("usage", {})
                    yield StreamDelta(
                        finish_reason=delta.get("stop_reason"),
                        usage=Usage(
                            completion_tokens=usage.get("output_tokens", 0),
                        )
                        if usage
                        else None,
                    )

                elif event_type == "message_start":
                    msg = data.get("message", {})
                    usage = msg.get("usage", {})
                    if usage:
                        yield StreamDelta(
                            usage=Usage(prompt_tokens=usage.get("input_tokens", 0))
                        )

    def _check_error(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        if resp.status_code == 401:
            raise AuthenticationError(
                "Invalid Anthropic API key", provider=self.name, status_code=401
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "Anthropic rate limit exceeded", provider=self.name, status_code=429
            )
        raise ProviderError(
            f"Anthropic API error {resp.status_code}: {resp.text}",
            provider=self.name,
            status_code=resp.status_code,
        )

    async def close(self) -> None:
        await self._client.aclose()
