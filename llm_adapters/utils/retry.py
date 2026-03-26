"""Exponential backoff retry utility."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from llm_adapters.exceptions import LLMError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[type[Exception], ...] = (RateLimitError,),
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        fn: Async callable to execute
        max_retries: Maximum number of retries (0 = no retry)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        retryable: Exception types that trigger a retry

    Returns:
        The result of fn()

    Raises:
        The last exception if all retries are exhausted
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable as e:
            last_error = e
            if attempt == max_retries:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "Retry %d/%d after %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    assert last_error is not None  # noqa: S101
    raise last_error
