"""Ollama local REST API adapter.

Supports: Any model pulled into Ollama (llama3, mistral, codellama, etc.)
API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llm_adapters.config import ProviderConfig
from llm_adapters.exceptions import ModelNotFoundError, ProviderError
from llm_adapters.models import ChatRequest, ChatResponse, StreamDelta, Usage
from llm_adapters.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama local model adapter."""

    name = "ollama"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def _build_payload(self, request: ChatRequest, stream: bool = False) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "stream": stream,
        }
        options: dict = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = request.stop
        if options:
            payload["options"] = options
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)
        resp = await self._client.post("/api/chat", json=payload)
        self._check_error(resp)
        data = resp.json()
        message = data.get("message", {})
        return ChatResponse(
            model=data.get("model", request.model),
            content=message.get("content", ""),
            finish_reason="stop" if data.get("done") else None,
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamDelta]:
        payload = self._build_payload(request, stream=True)
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            self._check_error(resp)
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                done = data.get("done", False)
                yield StreamDelta(
                    content=message.get("content", ""),
                    finish_reason="stop" if done else None,
                    usage=Usage(
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        total_tokens=(
                            data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        ),
                    )
                    if done
                    else None,
                )

    def _check_error(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Ollama model not found: {resp.text}", provider=self.name, status_code=404
            )
        raise ProviderError(
            f"Ollama error {resp.status_code}: {resp.text}",
            provider=self.name,
            status_code=resp.status_code,
        )

    async def close(self) -> None:
        await self._client.aclose()
