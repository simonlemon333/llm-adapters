"""OpenAI Chat Completions adapter.

Supports: GPT-4o, GPT-4, GPT-3.5-turbo, o1, o3, etc.
API docs: https://platform.openai.com/docs/api-reference/chat
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
from llm_adapters.models import ChatRequest, ChatResponse, StreamDelta, Usage
from llm_adapters.providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions API adapter."""

    name = "openai"

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

    def _build_payload(self, request: ChatRequest) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "stream": request.stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        if request.stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request.model_copy(update={"stream": False}))
        resp = await self._client.post("/chat/completions", json=payload)
        self._check_error(resp)
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=choice["message"]["content"] or "",
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamDelta]:
        payload = self._build_payload(request.model_copy(update={"stream": True}))
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            self._check_error(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                data = json.loads(data_str)
                choice = data["choices"][0] if data.get("choices") else None
                delta = choice["delta"] if choice else {}
                usage_data = data.get("usage")
                yield StreamDelta(
                    content=delta.get("content", ""),
                    finish_reason=choice.get("finish_reason") if choice else None,
                    usage=Usage(**usage_data) if usage_data else None,
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
            f"OpenAI API error {resp.status_code}: {resp.text}",
            provider=self.name,
            status_code=resp.status_code,
        )

    async def close(self) -> None:
        await self._client.aclose()
