"""Tests for retry utility."""

import pytest

from llm_adapters.exceptions import RateLimitError
from llm_adapters.utils.retry import with_retry


@pytest.mark.asyncio
async def test_retry_succeeds_first_try():
    call_count = 0

    async def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await with_retry(fn, max_retries=2, base_delay=0.01)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_failure():
    call_count = 0

    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError("rate limited", provider="test")
        return "ok"

    result = await with_retry(fn, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    async def fn():
        raise RateLimitError("rate limited", provider="test")

    with pytest.raises(RateLimitError):
        await with_retry(fn, max_retries=2, base_delay=0.01)
