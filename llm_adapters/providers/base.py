"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from llm_adapters.config import ProviderConfig
from llm_adapters.models import ChatRequest, ChatResponse, StreamDelta


class BaseLLMProvider(ABC):
    """Every adapter implements this interface — chat + stream, nothing more."""

    name: str  # e.g. "openai", "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the full response."""

    @abstractmethod
    def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamDelta]:
        """Send a chat request and yield streaming deltas."""
        ...  # pragma: no cover

    async def close(self) -> None:  # noqa: B027
        """Clean up resources. Override if your adapter holds a persistent client."""
