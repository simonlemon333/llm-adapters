"""Tests for LLMClient."""

import pytest

from llm_adapters.client import LLMClient
from llm_adapters.exceptions import LLMError, ProviderNotFoundError


def test_parse_model_valid():
    provider, model = LLMClient.parse_model("openai/gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"


def test_parse_model_with_nested_slash():
    provider, model = LLMClient.parse_model("ollama/llama3:70b")
    assert provider == "ollama"
    assert model == "llama3:70b"


def test_parse_model_invalid():
    with pytest.raises(LLMError, match="Invalid model format"):
        LLMClient.parse_model("gpt-4o")


def test_parse_model_case_insensitive():
    provider, model = LLMClient.parse_model("OpenAI/gpt-4o")
    assert provider == "openai"


@pytest.mark.asyncio
async def test_unknown_provider():
    client = LLMClient()
    with pytest.raises(ProviderNotFoundError, match="Unknown provider"):
        await client.chat(
            model="foobar/some-model",
            messages=[{"role": "user", "content": "Hi"}],
        )
