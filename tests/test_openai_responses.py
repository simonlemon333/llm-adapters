"""Tests for OpenAI Responses API adapter."""

import json

import pytest

from llm_adapters.config import ProviderConfig
from llm_adapters.exceptions import AuthenticationError, RateLimitError
from llm_adapters.models import ChatRequest, Message, Role
from llm_adapters.providers.openai_responses import OpenAIResponsesProvider


@pytest.fixture
def config():
    return ProviderConfig(api_key="test-key", base_url="https://api.openai.com")


@pytest.fixture
def request_with_system():
    return ChatRequest(
        model="gpt-4.1",
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ],
    )


@pytest.fixture
def simple_request():
    return ChatRequest(
        model="gpt-4.1",
        messages=[Message(role=Role.USER, content="Hello")],
    )


@pytest.mark.asyncio
async def test_responses_chat(config, simple_request, httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        json={
            "id": "resp-123",
            "model": "gpt-4.1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hi there!"}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            },
        },
    )
    provider = OpenAIResponsesProvider(config)
    resp = await provider.chat(simple_request)
    assert resp.content == "Hi there!"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 8
    assert resp.id == "resp-123"
    await provider.close()


@pytest.mark.asyncio
async def test_responses_system_becomes_developer(config, request_with_system, httpx_mock):
    """Verify system messages are converted to developer role."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        json={
            "id": "resp-456",
            "model": "gpt-4.1",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
        },
    )
    provider = OpenAIResponsesProvider(config)
    await provider.chat(request_with_system)

    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert sent["input"][0]["role"] == "developer"
    assert sent["input"][0]["content"] == "You are helpful."
    assert sent["input"][1]["role"] == "user"
    await provider.close()


@pytest.mark.asyncio
async def test_responses_max_tokens_maps_to_max_output_tokens(config, httpx_mock):
    """max_tokens should become max_output_tokens in Responses API."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        json={
            "id": "resp-789",
            "model": "gpt-4.1",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        },
    )
    provider = OpenAIResponsesProvider(config)
    req = ChatRequest(
        model="gpt-4.1",
        messages=[Message(role=Role.USER, content="Hi")],
        max_tokens=100,
    )
    await provider.chat(req)

    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert sent["max_output_tokens"] == 100
    assert "max_tokens" not in sent
    await provider.close()


@pytest.mark.asyncio
async def test_responses_auth_error(config, simple_request, httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        status_code=401,
        text="Unauthorized",
    )
    provider = OpenAIResponsesProvider(config)
    with pytest.raises(AuthenticationError):
        await provider.chat(simple_request)
    await provider.close()


@pytest.mark.asyncio
async def test_responses_rate_limit(config, simple_request, httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        status_code=429,
        text="Rate limited",
    )
    provider = OpenAIResponsesProvider(config)
    with pytest.raises(RateLimitError):
        await provider.chat(simple_request)
    await provider.close()
