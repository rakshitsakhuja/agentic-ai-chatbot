"""
LLM Observability — Langfuse v3 Tracer (correct API)
──────────────────────────────────────────────────────
Verified against langfuse==3.14.4

Key v3 rules:
  - start_as_current_span(name=...)        keyword-only, context manager
  - start_observation(name=..., as_type=)  replaces start_generation/start_span
  - .update(output=..., metadata=...)      set data BEFORE ending
  - .end()                                 no args except optional end_time
  - update_current_trace(session_id=...)   updates the active trace metadata
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from langfuse import Langfuse

from llm.base import ToolCall
from tools.registry import ToolResult


# ── Cost table (USD per 1M tokens) ───────────────────────────────────────────

COST_TABLE = {
    "claude-haiku-4-5-20251001": (0.80,   4.00),
    "claude-sonnet-4-6":         (3.00,  15.00),
    "claude-opus-4-6":           (15.00, 75.00),
    "gpt-4o-mini":               (0.15,   0.60),
    "gpt-4o":                    (2.50,  10.00),
    "llama-3.1-8b-instant":      (0.05,   0.08),
    "llama-3.3-70b-versatile":   (0.59,   0.79),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = COST_TABLE.get(model, (1.0, 3.0))
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


# ── Trace result ──────────────────────────────────────────────────────────────

@dataclass
class AgentTrace:
    trace_id: str
    session_id: str
    question: str
    answer: str
    model: str
    provider: str
    llm_calls: int
    tool_calls: int
    tool_errors: int
    answered_directly: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int

    def summary(self) -> str:
        mode = "DIRECT" if self.answered_directly else f"FRAMEWORK ({self.tool_calls} tools)"
        return (
            f"trace={self.trace_id[:8]} | mode={mode} | "
            f"llm_calls={self.llm_calls} | "
            f"tokens={self.prompt_tokens}+{self.completion_tokens} | "
            f"cost=${self.cost_usd:.5f} | latency={self.latency_ms}ms"
        )


# ── Tracer ────────────────────────────────────────────────────────────────────

class AgentTracer:
    def __init__(
        self,
        session_id: str = "default",
        model: str = "unknown",
        provider: str = "unknown",
        public_key: str = None,
        secret_key: str = None,
        host: str = "https://cloud.langfuse.com",
    ):
        self.session_id = session_id
        self.model = model
        self.provider = provider

        self._lf = Langfuse(
            public_key=public_key or os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=secret_key or os.environ["LANGFUSE_SECRET_KEY"],
            host=host,
        )

        # Per-turn state
        self._obs: Dict[str, Any] = {}       # tool_call_id → observation
        self._tool_starts: Dict[str, float] = {}
        self._llm_call_start: float = 0.0

    # ── Main API ──────────────────────────────────────────────────────────────

    def run(self, agent, question: str, stop_event=None) -> tuple:
        """Run agent inside a Langfuse trace. Returns (AgentResult, AgentTrace)."""
        self._obs = {}
        self._tool_starts = {}
        self._llm_call_start = time.time()
        start = time.time()

        with self._lf.start_as_current_span(name="agent_turn") as root:
            self._lf.update_current_trace(
                session_id=self.session_id,
                input={"question": question},
                metadata={"model": self.model, "provider": self.provider},
            )

            result = agent.run(question, stop_event=stop_event)

            # Aggregate stats
            tool_calls        = sum(len(s.tool_calls)  for s in result.steps)
            tool_errors       = sum(sum(1 for tr in s.tool_results if tr.is_error) for s in result.steps)
            prompt_tokens     = result.total_tokens.get("prompt_tokens", 0)
            completion_tokens = result.total_tokens.get("completion_tokens", 0)
            llm_calls         = len(result.steps) + 1

            # Update root span — .update() BEFORE .end() (end called by context manager)
            root.update(
                output={"answer": result.answer},
                usage_details={
                    "input":  prompt_tokens,
                    "output": completion_tokens,
                },
                metadata={
                    "llm_calls":         llm_calls,
                    "tool_calls":        tool_calls,
                    "tool_errors":       tool_errors,
                    "answered_directly": len(result.steps) == 0,
                    "success":           result.success,
                },
            )
            self._lf.update_current_trace(
                output={"answer": result.answer},
            )

        # Flush in background so it doesn't block returning the result to the UI.
        threading.Thread(target=self._lf.flush, daemon=True).start()

        latency_ms = int((time.time() - start) * 1000)
        trace = AgentTrace(
            trace_id          = str(uuid.uuid4()),
            session_id        = self.session_id,
            question          = question,
            answer            = result.answer,
            model             = self.model,
            provider          = self.provider,
            llm_calls         = llm_calls,
            tool_calls        = tool_calls,
            tool_errors       = tool_errors,
            answered_directly = len(result.steps) == 0,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            cost_usd          = estimate_cost(self.model, prompt_tokens, completion_tokens),
            latency_ms        = latency_ms,
        )
        return result, trace

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def make_callbacks(self) -> dict:
        return {
            "on_thought":     self._on_thought,
            "on_tool_call":   self._on_tool_call,
            "on_tool_result": self._on_tool_result,
            "on_final":       lambda a: None,
        }

    def _on_thought(self, iteration: int, thought: str):
        end_ts    = time.time()
        latency   = int((end_ts - self._llm_call_start) * 1000)
        llm_start = self._llm_call_start
        self._llm_call_start = end_ts

        # start_observation() always uses "now" as startTime — use
        # completion_start_time to record when the LLM call actually began.
        obs = self._lf.start_observation(
            name=f"llm_call_iter_{iteration}",
            as_type="generation",
            model=self.model,
            input={"iteration": iteration},
            completion_start_time=datetime.fromtimestamp(llm_start, tz=timezone.utc),
        )
        obs.update(
            output=thought or "(tool calls only)",
            metadata={"latency_ms": latency, "iteration": iteration},
        )
        obs.end(end_time=int(end_ts * 1000))

    def _on_tool_call(self, iteration: int, tc: ToolCall):
        self._tool_starts[tc.id] = time.time()
        obs = self._lf.start_observation(
            name=f"tool_{tc.name}",
            as_type="tool",
            input=tc.arguments,
            metadata={"iteration": iteration},
        )
        self._obs[tc.id] = obs

    def _on_tool_result(self, iteration: int, result: ToolResult):
        latency = int((time.time() - self._tool_starts.pop(result.tool_call_id, time.time())) * 1000)
        obs = self._obs.pop(result.tool_call_id, None)
        if obs:
            obs.update(
                output=result.content[:1000],
                level="ERROR" if result.is_error else "DEFAULT",
                metadata={"latency_ms": latency, "is_error": result.is_error},
            )
            obs.end()
        self._llm_call_start = time.time()
