"""
Orchestrator  (Multi-Agent Pattern)
──────────────────────────────────────
Routes tasks to specialized sub-agents based on task type.

Pattern:
  User task → Orchestrator → picks best sub-agent → runs it → returns result

Use when:
  - Different tasks need different tools / system prompts / models
  - You want domain isolation (e.g. data agent vs execution agent vs risk agent)
  - You want to run agents in parallel for independent subtasks

For trading use-case, sub-agents might be:
  - "data_agent"      → fetches and processes on/off-chain data
  - "analysis_agent"  → models market structure, regime, sentiment
  - "decision_agent"  → makes and sizes trade decisions
  - "execution_agent" → executes trades via exchange APIs
  - "risk_agent"      → enforces risk rules, stops, capital limits

Usage:
    orchestrator = Orchestrator(router_llm)
    orchestrator.register("data", data_agent, "Fetches and processes market data")
    orchestrator.register("risk", risk_agent, "Evaluates risk and enforces limits")

    result = orchestrator.run("What is the current BTC funding rate?")
    results = orchestrator.run_parallel(["Fetch BTC price", "Fetch ETH price"])
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm.base import BaseLLM, Message


ROUTER_SYSTEM = """You are a task router. Given a task, select the most appropriate agent to handle it.

Available agents:
{agent_descriptions}

Respond ONLY with a JSON object:
{{"agent": "<agent_name>", "reason": "<one line reason>"}}

If no agent is clearly best, pick the most general one."""


@dataclass
class SubAgentConfig:
    name: str
    agent: object        # ReActAgent instance
    description: str
    tags: List[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, router_llm: BaseLLM):
        self.router_llm = router_llm
        self._agents: Dict[str, SubAgentConfig] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, agent, description: str, tags: Optional[List[str]] = None):
        """Register a sub-agent."""
        self._agents[name] = SubAgentConfig(
            name=name,
            agent=agent,
            description=description,
            tags=tags or [],
        )

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, task: str) -> str:
        """Use LLM to decide which agent to use for a task."""
        if not self._agents:
            raise ValueError("No agents registered")
        if len(self._agents) == 1:
            return list(self._agents.keys())[0]

        descriptions = "\n".join(
            f"- {name}: {cfg.description}"
            for name, cfg in self._agents.items()
        )
        response = self.router_llm.chat(
            messages=[
                Message(role="system", content=ROUTER_SYSTEM.format(agent_descriptions=descriptions)),
                Message(role="user", content=f"Task: {task}"),
            ],
            temperature=0.0,
        )
        raw = (response.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            data = json.loads(raw)
            chosen = data.get("agent", "")
            if chosen in self._agents:
                return chosen
        except Exception:
            pass
        return list(self._agents.keys())[0]

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, task: str, agent_name: Optional[str] = None, verbose: bool = True):
        """
        Run a single task.
        If agent_name is None, auto-routes to best agent.
        """
        name = agent_name or self.route(task)
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not registered. Available: {list(self._agents.keys())}")

        if verbose:
            print(f"[Orchestrator] Routing '{task[:60]}...' → {name}")

        return self._agents[name].agent.run(task)

    def run_parallel(
        self,
        tasks: List[str],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """
        Run multiple independent tasks in parallel.
        Each task is auto-routed to the best agent.
        Returns {task: AgentResult}.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.run, task, None, verbose): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results[task] = future.result()
                except Exception as e:
                    results[task] = str(e)
        return results

    def run_pipeline(self, tasks: List[str], verbose: bool = True) -> List:
        """
        Run tasks sequentially (pipeline), passing each result as context to the next.
        """
        results = []
        context = ""
        for task in tasks:
            full_task = task
            if context:
                full_task = f"{task}\n\nContext from previous step:\n{context}"
            result = self.run(full_task, verbose=verbose)
            results.append(result)
            context = result.answer[:500]
        return results

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())
