"""
Anthropic / Claude Provider
────────────────────────────
Works with all Claude models: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6

Usage:
    from llm.anthropic_llm import AnthropicLLM

    llm = AnthropicLLM(model="claude-haiku-4-5-20251001", api_key="sk-ant-...")
"""

import json
from typing import Dict, Iterator, List, Optional

from .base import BaseLLM, LLMResponse, Message, ToolCall


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _split_system(self, messages: List[Message]):
        """Extract system message and return (system_str, other_messages)."""
        system = None
        rest = []
        for m in messages:
            if m.role == "system":
                system = m.content if isinstance(m.content, str) else json.dumps(m.content)
            else:
                rest.append(m)
        return system, rest

    def _to_anthropic_messages(self, messages: List[Message]) -> List[Dict]:
        result = []
        for m in messages:
            if m.role == "tool":
                # Tool result — append as user message with tool_result block
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                    }],
                })
            elif m.role == "assistant" and isinstance(m.content, list):
                # Already formatted (e.g. with tool_use blocks)
                result.append({"role": "assistant", "content": m.content})
            else:
                result.append({
                    "role": m.role,
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                })
        return result

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        system, rest = self._split_system(messages)
        ant_messages = self._to_anthropic_messages(rest)

        call_kwargs = dict(
            model=self.model,
            messages=ant_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = tools  # Anthropic uses input_schema directly

        response = self._client.messages.create(**call_kwargs)

        content_text = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        stop_reason = "stop"
        if response.stop_reason == "tool_use":
            stop_reason = "tool_use"
        elif response.stop_reason == "max_tokens":
            stop_reason = "length"

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Iterator[str]:
        system, rest = self._split_system(messages)
        ant_messages = self._to_anthropic_messages(rest)

        call_kwargs = dict(
            model=self.model,
            messages=ant_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            call_kwargs["system"] = system

        with self._client.messages.stream(**call_kwargs) as stream:
            for text in stream.text_stream:
                yield text
