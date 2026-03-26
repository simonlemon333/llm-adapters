"""Unified data models for LLM requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str = ""
    model: str = ""
    content: str = ""
    role: Role = Role.ASSISTANT
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class StreamDelta(BaseModel):
    content: str = ""
    role: Role | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
