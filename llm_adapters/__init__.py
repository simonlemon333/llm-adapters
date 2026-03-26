"""llm-adapters: A minimal, auditable LLM API library."""

from llm_adapters.client import LLMClient
from llm_adapters.models import ChatRequest, ChatResponse, Message, StreamDelta, Usage

__all__ = [
    "LLMClient",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "StreamDelta",
    "Usage",
]

__version__ = "0.1.1"
