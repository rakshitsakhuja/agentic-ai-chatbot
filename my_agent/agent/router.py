"""
TaskRouter — Automatically picks the right agentic pattern for a task.

Analyzes the task and returns one of:
  - "react"         → single ReAct loop (simple/medium tasks)
  - "plan_execute"  → Planner + ReActAgent (complex multi-step)
  - "reflect"       → ReAct + Reflection (high-stakes, quality-critical)
  - "orchestrate"   → Orchestrator (multi-domain, parallel subtasks)

The router itself uses a fast/cheap LLM call (1 round trip).
You can also skip the LLM and use rule-based routing (faster, free).
"""

import re
from dataclasses import dataclass
from typing import Optional

from llm.base import BaseLLM, Message


# ── Decision output ───────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    pattern: str           # "react" | "plan_execute" | "reflect" | "orchestrate"
    reason: str
    confidence: int        # 1-10
    use_reflection: bool   # can be stacked on top of any pattern


# ── Rule-based router (free, instant, no LLM call) ───────────────────────────

# Keywords that signal each pattern
_PLAN_SIGNALS = [
    "step by step", "first.*then", "pipeline", "workflow", "sequence",
    "multiple steps", "plan", "roadmap", "phase", "stage",
    "build and then", "after that", "finally",
]

_REFLECT_SIGNALS = [
    "accurate", "correct", "verify", "validate", "double.check",
    "high.stakes", "critical", "important", "must be right",
    "financial", "legal", "medical", "trade", "invest", "risk",
    "production", "deploy",
]

_ORCHESTRATE_SIGNALS = [
    "parallel", "simultaneously", "at the same time", "multiple agents",
    "different domains", "data.*and.*execute", "fetch.*and.*analyse",
    "coordinate", "delegate",
]

_COMPLEX_SIGNALS = [
    "analyse", "analyze", "research", "investigate", "comprehensive",
    "full report", "end.to.end", "complete", "everything about",
    "detailed", "in.depth",
]


def route_by_rules(task: str) -> RoutingDecision:
    """
    Fast rule-based routing. No LLM call. Use this for production
    where latency matters or when task types are predictable.
    """
    t = task.lower()

    # Score each pattern
    scores = {
        "plan_execute":  sum(1 for p in _PLAN_SIGNALS     if re.search(p, t)),
        "reflect":       sum(1 for p in _REFLECT_SIGNALS  if re.search(p, t)),
        "orchestrate":   sum(1 for p in _ORCHESTRATE_SIGNALS if re.search(p, t)),
        "complex":       sum(1 for p in _COMPLEX_SIGNALS  if re.search(p, t)),
    }

    use_reflection = scores["reflect"] > 0

    # Multi-step explicit → plan
    if scores["plan_execute"] >= 1:
        return RoutingDecision(
            pattern="plan_execute",
            reason="Task mentions sequential steps or workflow",
            confidence=8,
            use_reflection=use_reflection,
        )

    # Parallel/multi-domain → orchestrate
    if scores["orchestrate"] >= 1:
        return RoutingDecision(
            pattern="orchestrate",
            reason="Task involves parallel or multi-domain work",
            confidence=8,
            use_reflection=use_reflection,
        )

    # Complex but single domain → react (handles it fine)
    # Add reflection if stakes are high
    if scores["complex"] >= 2 or use_reflection:
        return RoutingDecision(
            pattern="react",
            reason="Complex task handled by ReAct; reflection added for quality",
            confidence=7,
            use_reflection=True,
        )

    # Default → plain ReAct
    return RoutingDecision(
        pattern="react",
        reason="Simple/direct task — ReAct is sufficient",
        confidence=9,
        use_reflection=False,
    )


# ── LLM-based router (smarter, costs 1 cheap LLM call) ───────────────────────

ROUTER_SYSTEM = """You are an agentic task classifier. Given a task, decide the best execution pattern.

Patterns:
- react         : Single iterative loop. Best for: direct questions, simple actions, lookup tasks, anything completable in <5 tool calls.
- plan_execute  : Break into subtasks first, then execute each. Best for: multi-step workflows, "do X then Y then Z", reports, pipelines.
- reflect       : Execute then critique and retry. Best for: high-stakes outputs, financial decisions, anything where accuracy is critical.
- orchestrate   : Route to specialised sub-agents. Best for: tasks needing different tools/domains, parallel independent subtasks.

Note: reflect can be combined with any pattern (set use_reflection=true).

Respond ONLY with JSON:
{
  "pattern": "react|plan_execute|reflect|orchestrate",
  "reason": "one sentence",
  "confidence": 1-10,
  "use_reflection": true|false
}"""


def route_by_llm(task: str, llm: BaseLLM) -> RoutingDecision:
    """
    LLM-based routing. Smarter for ambiguous tasks.
    Costs 1 cheap LLM call (use haiku/gpt-4o-mini).
    """
    import json

    response = llm.chat(
        messages=[
            Message(role="system", content=ROUTER_SYSTEM),
            Message(role="user", content=f"Task: {task}"),
        ],
        temperature=0.0,
        max_tokens=200,
    )

    raw = (response.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

    try:
        d = json.loads(raw)
        return RoutingDecision(
            pattern=d.get("pattern", "react"),
            reason=d.get("reason", ""),
            confidence=int(d.get("confidence", 7)),
            use_reflection=bool(d.get("use_reflection", False)),
        )
    except Exception:
        return route_by_rules(task)  # fallback to rules


# ── Smart Agent — wraps everything, auto-selects pattern ─────────────────────

class SmartAgent:
    """
    Drop-in agent that automatically picks the right pattern per task.

    Usage:
        smart = SmartAgent(llm, tools, routing="rules")  # or "llm"
        result = smart.run("Analyse BTC market and place a trade if conditions are right")
        # → automatically uses plan_execute + reflection
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools,                          # ToolRegistry
        memory=None,                    # ShortTermMemory (optional)
        orchestrator=None,              # Orchestrator (optional, for orchestrate pattern)
        routing: str = "rules",         # "rules" (fast) or "llm" (smart)
        system_prompt: str = "",
        max_iterations: int = 15,
        verbose: bool = True,
    ):
        from agent.react import ReActAgent
        from agent.planner import Planner, PlanExecutor
        from agent.reflection import Reflector
        from memory.short_term import ShortTermMemory

        self.llm = llm
        self.routing = routing
        self.verbose = verbose
        self.orchestrator = orchestrator

        # Build core agent
        self._memory = memory or ShortTermMemory()
        self._agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=self._memory,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
        )
        self._planner = Planner(llm)
        self._executor = PlanExecutor(self._agent)
        self._reflector = Reflector(llm, max_retries=2)

    def route(self, task: str) -> RoutingDecision:
        if self.routing == "llm":
            return route_by_llm(task, self.llm)
        return route_by_rules(task)

    def run(self, task: str):
        decision = self.route(task)

        if self.verbose:
            print(f"\n[Router] pattern={decision.pattern} | reflect={decision.use_reflection} | confidence={decision.confidence}/10")
            print(f"[Router] reason: {decision.reason}")

        # ── Execute chosen pattern ────────────────────────────────────────────

        if decision.pattern == "orchestrate" and self.orchestrator:
            result = self.orchestrator.run(task, verbose=self.verbose)

        elif decision.pattern == "plan_execute":
            plan = self._planner.create_plan(task)
            if self.verbose:
                print(f"\n[Planner] {len(plan.steps)} steps:")
                for s in plan.steps:
                    print(f"  {s.step}. {s.title}")
            completed = self._executor.execute(plan)
            # Synthesise final answer from all step results
            synthesis_task = (
                f"Original goal: {task}\n\n"
                f"Steps completed:\n" +
                "\n".join(f"{s.step}. {s.title}: {s.result}" for s in completed.steps)
                + "\n\nSummarise the final answer."
            )
            self._agent.reset()
            result = self._agent.run(synthesis_task)

        else:
            # Default: react
            result = self._agent.run(task)

        # ── Optionally add reflection on top ─────────────────────────────────
        if decision.use_reflection and decision.pattern != "plan_execute":
            if self.verbose:
                print("\n[Reflection] Running quality check...")
            reflection = self._reflector.reflect_and_refine(
                self._agent, task, result, verbose=self.verbose
            )
            # Wrap result to expose .answer
            result.answer = reflection.final_answer

        return result
