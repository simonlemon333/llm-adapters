"""Unified exception hierarchy for llm-adapters."""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all llm-adapters errors."""

    def __init__(self, message: str, *, provider: str = "", status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(LLMError):
    """Invalid or missing API key."""


class RateLimitError(LLMError):
    """Provider rate limit exceeded."""


class ModelNotFoundError(LLMError):
    """Requested model does not exist."""


class ProviderNotFoundError(LLMError):
    """Provider name not recognized."""


class ProviderError(LLMError):
    """Provider returned an unexpected error."""


class TimeoutError(LLMError):
    """Request timed out."""


class StreamError(LLMError):
    """Error during streaming response."""
