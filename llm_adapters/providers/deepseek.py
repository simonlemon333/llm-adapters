"""DeepSeek adapter — uses OpenAI-compatible API.

Supports: deepseek-chat, deepseek-coder, etc.
API docs: https://platform.deepseek.com/api-docs
"""

from __future__ import annotations

from llm_adapters.config import ProviderConfig
from llm_adapters.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek uses an OpenAI-compatible API, so we inherit directly."""

    name = "deepseek"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        # Override error messages to say "DeepSeek" instead of "OpenAI"

    def _check_error(self, resp):  # type: ignore[override]
        if resp.status_code < 400:
            return
        from llm_adapters.exceptions import (
            AuthenticationError,
            ProviderError,
            RateLimitError,
        )

        if resp.status_code == 401:
            raise AuthenticationError(
                "Invalid DeepSeek API key", provider=self.name, status_code=401
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "DeepSeek rate limit exceeded", provider=self.name, status_code=429
            )
        raise ProviderError(
            f"DeepSeek API error {resp.status_code}: {resp.text}",
            provider=self.name,
            status_code=resp.status_code,
        )
