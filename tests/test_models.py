"""Tests for data models."""

from llm_adapters.models import ChatRequest, ChatResponse, Message, Role, StreamDelta, Usage


def test_message_creation():
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"


def test_chat_request_defaults():
    req = ChatRequest(
        model="gpt-4o",
        messages=[Message(role=Role.USER, content="Hi")],
    )
    assert req.temperature is None
    assert req.stream is False
    assert req.extra == {}


def test_chat_response_defaults():
    resp = ChatResponse()
    assert resp.content == ""
    assert resp.usage.total_tokens == 0
    assert resp.role == Role.ASSISTANT


def test_stream_delta():
    delta = StreamDelta(content="Hello")
    assert delta.content == "Hello"
    assert delta.finish_reason is None


def test_usage_total():
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert usage.total_tokens == 30
