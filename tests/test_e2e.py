"""E2E tests against real APIs via OpenRouter.

Run with: pytest tests/test_e2e.py -v
Requires OPENROUTER_API_KEY in environment.
"""

import os

import pytest

from llm_adapters.client import LLMClient
from llm_adapters.config import ProviderConfig

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

skip_no_key = pytest.mark.skipif(not OPENROUTER_KEY, reason="OPENROUTER_API_KEY not set")


@pytest.fixture
def client():
    """Client configured to use OpenRouter as an OpenAI-compatible backend."""
    return LLMClient(
        openai=ProviderConfig(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE),
    )


@skip_no_key
@pytest.mark.asyncio
async def test_e2e_chat(client):
    """Test a real chat completion via OpenRouter (cheapest model)."""
    resp = await client.chat(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=20,
    )
    assert resp.content.strip().lower().startswith("hello")
    assert resp.usage.total_tokens > 0
    print(f"\n  Response: {resp.content!r}")
    print(f"  Tokens: {resp.usage.total_tokens}")
    await client.close()


@skip_no_key
@pytest.mark.asyncio
async def test_e2e_stream(client):
    """Test real streaming via OpenRouter."""
    chunks = []
    async for delta in client.chat_stream(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Count from 1 to 3."}],
        max_tokens=20,
    ):
        if delta.content:
            chunks.append(delta.content)
    full = "".join(chunks)
    assert "1" in full and "2" in full and "3" in full
    print(f"\n  Streamed: {full!r}")
    await client.close()
