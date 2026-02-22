"""
LLM Abstraction Layer
─────────────────────
Defines the universal interface all LLM providers must implement.
Add any new provider by subclassing BaseLLM and implementing `chat()`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ToolCall:
    """A single tool/function call requested by the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: Optional[str]                  # text output (None if only tool calls)
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"               # "stop" | "tool_use" | "length" | "error"
    usage: Dict[str, int] = field(default_factory=dict)  # prompt/completion tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_final(self) -> bool:
        return not self.has_tool_calls


@dataclass
class Message:
    """A single message in a conversation."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: Any       # str for text, list for multi-part content
    tool_call_id: Optional[str] = None   # only for role="tool"
    name: Optional[str] = None           # tool name for role="tool"
    tool_calls: List["ToolCall"] = field(default_factory=list)  # outgoing calls from assistant


class BaseLLM(ABC):
    """
    Abstract base for all LLM providers.

    To add a new provider:
    1. Subclass BaseLLM
    2. Implement chat()
    3. Optionally implement stream_chat()
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,  # universal tool schema list
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Send messages and get a response. Block until complete."""
        ...

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Iterator[str]:
        """Stream text tokens. Default: calls chat() and yields content."""
        response = self.chat(messages, tools, temperature, max_tokens, **kwargs)
        if response.content:
            yield response.content
