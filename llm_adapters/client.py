"""Unified LLM client — the main entry point for llm-adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator

from llm_adapters.config import ProviderConfig
from llm_adapters.exceptions import LLMError
from llm_adapters.models import ChatRequest, ChatResponse, Message, Role, StreamDelta
from llm_adapters.providers.base import BaseLLMProvider
from llm_adapters.providers.registry import ProviderRegistry


class LLMClient:
    """Unified client that routes requests to the correct provider.

    Usage:
        client = LLMClient()
        response = await client.chat(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(self, **provider_configs: ProviderConfig) -> None:
        """Initialize the client.

        Args:
            **provider_configs: Optional per-provider configs, keyed by provider name.
                e.g. LLMClient(openai=ProviderConfig(api_key="sk-..."))
        """
        self._registry = ProviderRegistry()
        self._provider_configs = provider_configs

    @staticmethod
    def parse_model(model: str) -> tuple[str, str]:
        """Parse 'provider/model' string into (provider_name, model_name).

        Raises LLMError if the format is invalid.
        """
        if "/" not in model:
            raise LLMError(
                f"Invalid model format: {model!r}. Expected 'provider/model' "
                f"(e.g. 'openai/gpt-4o', 'anthropic/claude-sonnet-4-20250514')."
            )
        provider, model_name = model.split("/", 1)
        return provider.lower(), model_name

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, str] | Message],
    ) -> list[Message]:
        """Accept both dicts and Message objects."""
        result = []
        for m in messages:
            if isinstance(m, Message):
                result.append(m)
            else:
                result.append(Message(role=Role(m["role"]), content=m["content"]))
        return result

    def _get_provider(self, provider_name: str) -> BaseLLMProvider:
        config = self._provider_configs.get(provider_name)
        return self._registry.get(provider_name, config)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str] | Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> ChatResponse:
        """Send a chat request to the appropriate provider.

        Args:
            model: Provider/model string, e.g. "openai/gpt-4o"
            messages: List of message dicts or Message objects
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
        """
        provider_name, model_name = self.parse_model(model)
        provider = self._get_provider(provider_name)
        request = ChatRequest(
            model=model_name,
            messages=self._normalize_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        return await provider.chat(request)

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str] | Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Send a streaming chat request.

        Args:
            model: Provider/model string, e.g. "openai/gpt-4o"
            messages: List of message dicts or Message objects
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Yields:
            StreamDelta objects with incremental content
        """
        provider_name, model_name = self.parse_model(model)
        provider = self._get_provider(provider_name)
        request = ChatRequest(
            model=model_name,
            messages=self._normalize_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=True,
        )
        async for delta in provider.chat_stream(request):
            yield delta

    async def close(self) -> None:
        """Close all provider connections."""
        await self._registry.close_all()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
