"""Tests for provider adapters using mocked httpx responses."""

import json

import httpx
import pytest

from llm_adapters.config import ProviderConfig
from llm_adapters.exceptions import AuthenticationError, RateLimitError
from llm_adapters.models import ChatRequest, Message, Role
from llm_adapters.providers.openai import OpenAIProvider
from llm_adapters.providers.anthropic import AnthropicProvider
from llm_adapters.providers.ollama import OllamaProvider


# --- Fixtures ---

@pytest.fixture
def openai_config():
    return ProviderConfig(api_key="test-key", base_url="https://api.openai.com/v1")


@pytest.fixture
def anthropic_config():
    return ProviderConfig(api_key="test-key", base_url="https://api.anthropic.com")


@pytest.fixture
def ollama_config():
    return ProviderConfig(base_url="http://localhost:11434")


@pytest.fixture
def sample_request():
    return ChatRequest(
        model="test-model",
        messages=[Message(role=Role.USER, content="Hello")],
    )


# --- OpenAI Tests ---

@pytest.mark.asyncio
async def test_openai_chat(openai_config, sample_request, httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "id": "chatcmpl-123",
            "model": "test-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
    )
    provider = OpenAIProvider(openai_config)
    resp = await provider.chat(sample_request)
    assert resp.content == "Hi there!"
    assert resp.usage.total_tokens == 8
    assert resp.finish_reason == "stop"
    await provider.close()


@pytest.mark.asyncio
async def test_openai_auth_error(openai_config, sample_request, httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        status_code=401,
        text="Unauthorized",
    )
    provider = OpenAIProvider(openai_config)
    with pytest.raises(AuthenticationError):
        await provider.chat(sample_request)
    await provider.close()


@pytest.mark.asyncio
async def test_openai_rate_limit(openai_config, sample_request, httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        status_code=429,
        text="Rate limited",
    )
    provider = OpenAIProvider(openai_config)
    with pytest.raises(RateLimitError):
        await provider.chat(sample_request)
    await provider.close()


# --- Anthropic Tests ---

@pytest.mark.asyncio
async def test_anthropic_chat(anthropic_config, httpx_mock):
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json={
            "id": "msg-123",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    provider = AnthropicProvider(anthropic_config)
    request = ChatRequest(
        model="claude-sonnet-4-20250514",
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ],
    )
    resp = await provider.chat(request)
    assert resp.content == "Hello from Claude!"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5

    # Verify system message was separated
    sent_request = json.loads(httpx_mock.get_requests()[0].content)
    assert sent_request["system"] == "You are helpful."
    assert all(m["role"] != "system" for m in sent_request["messages"])
    await provider.close()


# --- Ollama Tests ---

@pytest.mark.asyncio
async def test_ollama_chat(ollama_config, sample_request, httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "message": {"role": "assistant", "content": "Local model response"},
            "done": True,
            "prompt_eval_count": 8,
            "eval_count": 12,
        },
    )
    provider = OllamaProvider(ollama_config)
    resp = await provider.chat(sample_request)
    assert resp.content == "Local model response"
    assert resp.usage.prompt_tokens == 8
    assert resp.usage.completion_tokens == 12
    assert resp.finish_reason == "stop"
    await provider.close()
