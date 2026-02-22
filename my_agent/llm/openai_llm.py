"""
OpenAI-Compatible Provider
──────────────────────────
Works with: OpenAI, Azure OpenAI, Groq, Ollama, Together AI,
            DeepSeek, Mistral, Perplexity, LM Studio, Anyscale —
            anything that speaks the OpenAI API format.

Usage:
    from llm.openai_llm import OpenAILLM

    # OpenAI
    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-...")

    # Groq (fast inference)
    llm = OpenAILLM(model="llama-3.1-8b-instant", api_key="gsk_...",
                     base_url="https://api.groq.com/openai/v1")

    # Ollama (local)
    llm = OpenAILLM(model="llama3.2", base_url="http://localhost:11434/v1", api_key="ollama")

    # DeepSeek
    llm = OpenAILLM(model="deepseek-chat", api_key="...",
                     base_url="https://api.deepseek.com/v1")
"""

import json
import re
import uuid
from typing import Dict, Iterator, List, Optional

from .base import BaseLLM, LLMResponse, Message, ToolCall


# ── Groq malformed-tool-call recovery ─────────────────────────────────────────
# Groq's llama models sometimes return <function=name{args}</function> instead
# of a proper JSON tool call.  The API rejects it with tool_use_failed (400).
# We intercept that error, parse the raw generation, and recover gracefully.

_GROQ_FUNC_RE = re.compile(r"<function=(\w+)(\{.*?\})", re.DOTALL)


def _recover_groq_tool_calls(failed_gen: str) -> List[ToolCall]:
    """Parse one or more <function=name{...}> patterns from a failed_generation string."""
    calls = []
    for name, raw_args in _GROQ_FUNC_RE.findall(failed_gen):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        calls.append(ToolCall(id=f"rc_{uuid.uuid4().hex[:8]}", name=name, arguments=args))
    return calls


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def _to_openai_messages(self, messages: List[Message]) -> List[Dict]:
        result = []
        for m in messages:
            if m.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                })
            elif m.role == "assistant" and m.tool_calls:
                # Assistant made tool calls — must include them so the API
                # can match tool results back to these call IDs
                msg: Dict = {"role": "assistant", "content": m.content}
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
                result.append(msg)
            else:
                result.append({
                    "role": m.role,
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                })
        return result

    def _to_openai_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert universal tool schema → OpenAI function schema."""
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return openai_tools

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        oai_messages = self._to_openai_messages(messages)
        call_kwargs = dict(
            model=self.model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if tools:
            call_kwargs["tools"] = self._to_openai_tools(tools)
            call_kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            # ── Groq tool_use_failed recovery ────────────────────────────────
            # When Groq's model outputs <function=name{args}</function> instead
            # of a proper JSON tool call the API returns 400 tool_use_failed.
            # We parse the failed_generation and recover the tool call.
            recovered = self._try_recover_groq(exc)
            if recovered:
                return recovered
            raise  # not a recoverable error — re-raise as-is

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        stop_reason = "stop"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "length"

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _try_recover_groq(self, exc: Exception) -> Optional[LLMResponse]:
        """
        If exc is a Groq tool_use_failed error, parse the failed_generation
        field and return a synthetic LLMResponse with the recovered tool calls.
        Returns None if the error is something else entirely.
        """
        try:
            from openai import BadRequestError
            if not isinstance(exc, BadRequestError):
                return None

            err_str = str(exc)
            if "tool_use_failed" not in err_str:
                return None

            failed_gen = None

            # Strategy 1: parse exc.body (dict from the SDK)
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                err_dict = body.get("error", body)   # handle both wrapped and flat
                if isinstance(err_dict, dict):
                    failed_gen = err_dict.get("failed_generation") or ""

            # Strategy 2: parse the string representation (always available)
            if not failed_gen:
                m = re.search(r"'failed_generation':\s*'(.+?)</function>'", err_str, re.DOTALL)
                if m:
                    failed_gen = m.group(1) + "</function>"

            if not failed_gen:
                print(f"[Groq recovery] tool_use_failed but no failed_generation found in error body.", flush=True)
                return None

            calls = _recover_groq_tool_calls(failed_gen)
            if not calls:
                print(f"[Groq recovery] Could not parse tool calls from: {failed_gen[:120]}", flush=True)
                return None

            print(
                "[Groq recovery] Recovered malformed tool call(s): "
                + ", ".join(f"{c.name}({c.arguments})" for c in calls),
                flush=True,
            )
            return LLMResponse(
                content=None,
                tool_calls=calls,
                stop_reason="tool_use",
                usage={},
            )
        except Exception as inner:
            print(f"[Groq recovery] Recovery itself raised: {inner}", flush=True)
            return None

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Iterator[str]:
        oai_messages = self._to_openai_messages(messages)
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
