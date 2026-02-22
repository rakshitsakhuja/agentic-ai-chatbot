"""
ReAct Agent  (Reasoning + Acting)
──────────────────────────────────
Core agentic loop:

  while not done:
      1. THINK  — LLM reasons about what to do next
      2. ACT    — LLM calls a tool (or returns final answer)
      3. OBSERVE — tool result is fed back into context
      repeat

The loop terminates when:
  - Model returns a response with no tool calls  → final answer
  - Max iterations reached                       → forced stop

Design notes:
  - Provider-agnostic: works with any BaseLLM implementation
  - Tool-agnostic: pass any ToolRegistry
  - Memory-integrated: uses ShortTermMemory for sliding context
  - Event hooks: on_thought, on_tool_call, on_tool_result, on_final
    → override or pass callbacks to observe agent behaviour in real time
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

from llm.base import BaseLLM, Message, ToolCall
from tools.registry import ToolRegistry, ToolResult
from tools.router import ToolRouter
from memory.short_term import ShortTermMemory


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    iteration: int
    thought: Optional[str]
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    elapsed_ms: int


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep]
    success: bool
    total_tokens: Dict[str, int]
    elapsed_ms: int

    def summary(self) -> str:
        lines = [
            f"Answer: {self.answer}",
            f"Steps: {len(self.steps)} | Tokens: {self.total_tokens} | Time: {self.elapsed_ms}ms",
        ]
        for s in self.steps:
            if s.tool_calls:
                calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in s.tool_calls)
                lines.append(f"  [{s.iteration}] Tools: {calls}")
        return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are a powerful AI agent that can use tools to answer questions and complete tasks.

Today's date: {current_date}
Use this date whenever the user refers to "today", "yesterday", "last 2 days", "this week", etc.
For arXiv date filters use YYYYMMDD format (e.g. {current_date_compact}).

When given a task:
1. Figure out what information or actions are needed.
2. Call the appropriate tool(s) with correct inputs.
3. Use the tool results to build your answer.
4. Repeat until you have everything you need.

Rules:
- Use tools whenever they would give a better answer (files, code, web, calculations).
- Always prefer specialized tools over generic ones (e.g. use search_arxiv, not http_request, for arXiv papers).
- Never use http_request on arxiv.org — it returns raw HTML. Use search_arxiv instead.
- If a tool returns an error, try a different approach or different inputs.
- When you have enough information, give a direct, clean answer.
- Be precise, concise, and factual. Do not make up information.
- Your final answer must be plain natural language — no labels like "ACT:", "OBSERVE:", "THOUGHT:", etc.

Tool creation guidance (when create_tool is available):
- ONE-OFF task (trivial code, user asks once): use python_repl directly. Do NOT create a tool.
- MISSING capability (no existing tool can do the job — e.g. http_request returns unusable HTML, need structured parsing, or the task clearly recurs): proactively call create_tool WITHOUT waiting for the user to ask, then immediately use the new tool to complete the task.
- REUSABLE capability (user says "save this", "add a tool for", "I'll need this again"): use create_tool to write and register it permanently.
- After creating a tool, always use it in the same response to complete the original task.

Naming rules — CRITICAL, strictly enforced:
- Name must describe WHAT THE TOOL DOES, not where it is used or what domain it targets.
  ✓ GOOD generic names: scrape_url, fetch_html_text, parse_markdown_table, extract_json_from_url
  ✗ BAD task-specific names: crawl_langchain_docs, get_github_readme, fetch_openai_pricing
- Before creating any tool, call list_dynamic_tools first. If a tool with similar functionality already exists (e.g. scrape_url covers any URL scraping), REUSE it — never create a near-duplicate.
- If an existing tool almost fits but needs a small tweak, call create_tool with the SAME name to overwrite it rather than making a new one.

{extra_instructions}"""


# ── ReAct Agent ───────────────────────────────────────────────────────────────

class ReActAgent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        memory: Optional[ShortTermMemory] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 15,
        max_tool_result_chars: int = 3000,  # truncate tool results in memory to limit context size
        tool_router: Optional[ToolRouter] = None,  # query-time tool selection
        # Callbacks — called at each stage (optional, for logging/UI/monitoring)
        on_thought: Optional[Callable[[int, str], None]] = None,
        on_tool_call: Optional[Callable[[int, ToolCall], None]] = None,
        on_tool_result: Optional[Callable[[int, ToolResult], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory or ShortTermMemory()
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.tool_router = tool_router

        # Callbacks
        self._on_thought = on_thought or self._default_thought
        self._on_tool_call = on_tool_call or self._default_tool_call
        self._on_tool_result = on_tool_result or self._default_tool_result
        self._on_final = on_final or self._default_final

        # Set system prompt
        today = date.today()
        extra = f"\nAdditional instructions:\n{system_prompt}" if system_prompt else ""
        self.memory.set_system(REACT_SYSTEM_PROMPT.format(
            current_date=today.strftime("%Y-%m-%d"),
            current_date_compact=today.strftime("%Y%m%d"),
            extra_instructions=extra,
        ))

        # Token tracking
        self._total_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        stop_event: Optional[threading.Event] = None,
    ) -> AgentResult:
        """Run the full ReAct loop for a task. Returns AgentResult.

        Args:
            task:        The user prompt / task description.
            stop_event:  Optional threading.Event. When set() the loop exits
                         gracefully after the current iteration finishes.
        """
        start = time.time()
        self.memory.add_user(task)
        steps: List[AgentStep] = []

        # Track consecutive identical failing tool calls to break retry loops.
        # Key: (tool_name, args_str), Value: consecutive error count
        _error_counts: Dict[tuple, int] = {}
        _MAX_SAME_ERROR = 2  # abort after this many identical errors in a row

        for i in range(1, self.max_iterations + 1):
            # ── User-requested stop ───────────────────────────────────────────
            if stop_event and stop_event.is_set():
                answer = "Processing was stopped."
                self._on_final(answer)
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    success=False,
                    total_tokens=dict(self._total_tokens),
                    elapsed_ms=int((time.time() - start) * 1000),
                )

            step_start = time.time()

            # ── Tool routing ──────────────────────────────────────────────────
            # Re-evaluated every iteration so newly created (dynamic) tools are
            # immediately visible to the model without waiting for the next run.
            if self.tool_router:
                _active_tools = self.tool_router.select(task, self.tools.list_tools())
            else:
                _active_tools = None

            tool_schema = self.tools.as_schema(names=_active_tools)

            response = self.llm.chat(
                messages=self.memory.get_messages(),
                tools=tool_schema if tool_schema else None,
            )

            # Track tokens
            for k in ("prompt_tokens", "completion_tokens"):
                self._total_tokens[k] = self._total_tokens.get(k, 0) + response.usage.get(k, 0)

            # Emit thought (always — so tracer gets a span for every LLM call)
            self._on_thought(i, response.content or "")

            # ── Final answer: no tool calls ──────────────────────────────────
            if response.is_final:
                answer = response.content or "(no response)"
                self._on_final(answer)
                self.memory.add_assistant(answer)
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    success=True,
                    total_tokens=dict(self._total_tokens),
                    elapsed_ms=int((time.time() - start) * 1000),
                )

            # ── Tool calls ───────────────────────────────────────────────────
            # Deduplicate: some models return the same tool call many times in
            # one response (parallel tool calling gone wrong). Keep only the
            # first occurrence of each (name, args) pair.
            seen_calls: set = set()
            unique_tool_calls = []
            for tc in response.tool_calls:
                key = (tc.name, str(sorted(tc.arguments.items())))
                if key not in seen_calls:
                    seen_calls.add(key)
                    unique_tool_calls.append(tc)

            # Also cap total tool calls per iteration to avoid runaway loops
            unique_tool_calls = unique_tool_calls[:5]

            # Store assistant message WITH tool_calls so the API can match
            # tool results back to the correct call IDs on the next turn
            self.memory.add_assistant(response.content or "", tool_calls=unique_tool_calls)

            tool_results: List[ToolResult] = []
            for tc in unique_tool_calls:
                self._on_tool_call(i, tc)
                result = self.tools.execute(tc.name, tc.id, tc.arguments)
                self._on_tool_result(i, result)
                tool_results.append(result)

                # ── Retry-loop guard ─────────────────────────────────────────
                err_key = (tc.name, str(sorted(tc.arguments.items())))
                if result.is_error:
                    _error_counts[err_key] = _error_counts.get(err_key, 0) + 1
                    if _error_counts[err_key] >= _MAX_SAME_ERROR:
                        # Inject a stern hint so the model stops retrying
                        abort_msg = (
                            f"[Agent] Tool '{tc.name}' has failed {_MAX_SAME_ERROR} times "
                            f"with the same arguments. Stop retrying this call. "
                            f"Either use a different tool, change the inputs, or tell the user "
                            f"you cannot complete this step."
                        )
                        self.memory.add_tool_result(
                            tool_call_id=tc.id + "_abort",
                            content=abort_msg,
                            tool_name="__agent__",
                        )
                else:
                    _error_counts.pop(err_key, None)  # reset on success

                # Truncate large results before storing in memory to prevent
                # context bloat (e.g. full paper text from arXiv RAG)
                mem_content = result.content
                if len(mem_content) > self.max_tool_result_chars:
                    mem_content = (
                        mem_content[:self.max_tool_result_chars]
                        + f"\n[...truncated — {len(result.content) - self.max_tool_result_chars} more chars]"
                    )
                self.memory.add_tool_result(
                    tool_call_id=tc.id,
                    content=mem_content,
                    tool_name=tc.name,
                )

            steps.append(AgentStep(
                iteration=i,
                thought=response.content,
                tool_calls=unique_tool_calls,
                tool_results=tool_results,
                elapsed_ms=int((time.time() - step_start) * 1000),
            ))

        # Max iterations hit — add a closing assistant message so memory never
        # ends with bare tool_results.  Without this, the next call to add_user()
        # creates an invalid sequence ([..., tool_result, user]) that the API
        # rejects with a 400 "tool message must follow tool_calls" error.
        answer = "Max iterations reached without completing the task."
        self._on_final(answer)
        self.memory.add_assistant(answer)
        return AgentResult(
            answer=answer,
            steps=steps,
            success=False,
            total_tokens=dict(self._total_tokens),
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def chat(self, message: str) -> str:
        """Convenience: run and return just the answer string."""
        return self.run(message).answer

    def reset(self):
        """Clear conversation history (keeps system prompt)."""
        self.memory.clear()
        self._total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

    # ── Default console callbacks ─────────────────────────────────────────────

    @staticmethod
    def _default_thought(i: int, thought: str):
        if not thought:
            return
        preview = thought[:200] + ("..." if len(thought) > 200 else "")
        print(f"\n[{i}] THOUGHT: {preview}")

    @staticmethod
    def _default_tool_call(i: int, tc: ToolCall):
        print(f"[{i}]  → CALL  {tc.name}({tc.arguments})")

    @staticmethod
    def _default_tool_result(i: int, result: ToolResult):
        status = "ERROR" if result.is_error else "OK"
        preview = result.content[:150].replace("\n", " ")
        ellipsis = "..." if len(result.content) > 150 else ""
        print(f"[{i}]  ← {status}   {preview}{ellipsis}")

    @staticmethod
    def _default_final(answer: str):
        print(f"\n{'─'*60}\nFINAL: {answer}\n{'─'*60}")
