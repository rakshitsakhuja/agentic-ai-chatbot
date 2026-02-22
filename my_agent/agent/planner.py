"""
Planner  (Task Decomposition Pattern)
──────────────────────────────────────
Breaks a complex goal into an ordered list of subtasks.
Each subtask is then executed by a ReActAgent.

Pattern: Plan → Execute Each Step → Aggregate Results

Use when:
  - Tasks are too complex for a single ReAct loop
  - Steps have dependencies (step 2 needs output of step 1)
  - You want to show the user a plan before execution
  - Long-horizon tasks (trading: fetch data → analyse → decide → execute)

Usage:
    planner = Planner(llm)
    plan = planner.create_plan("Build a trading signal for BTC/USDT")
    # plan.steps = [Step("Fetch price data"), Step("Compute indicators"), ...]

    executor = PlanExecutor(react_agent)
    result = executor.execute(plan)
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

from llm.base import BaseLLM, Message


PLANNER_SYSTEM = """You are a task planning expert. Given a complex goal, break it into clear, ordered, actionable subtasks.

Rules:
- Each subtask must be self-contained and achievable in isolation
- Order tasks so dependencies come first
- Be specific — vague steps like "analyse data" should say WHAT data and HOW
- Number each step
- Return ONLY a JSON array of step objects, no other text

Format:
[
  {"step": 1, "title": "Short title", "description": "Detailed what/how/why"},
  {"step": 2, ...}
]"""


@dataclass
class PlanStep:
    step: int
    title: str
    description: str
    result: Optional[str] = None
    success: Optional[bool] = None


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]

    def display(self) -> str:
        lines = [f"Goal: {self.goal}", f"Steps ({len(self.steps)}):"]
        for s in self.steps:
            status = ""
            if s.success is True:
                status = " ✓"
            elif s.success is False:
                status = " ✗"
            lines.append(f"  {s.step}. {s.title}{status}")
            lines.append(f"     {s.description}")
        return "\n".join(lines)


class Planner:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def create_plan(self, goal: str, context: str = "") -> Plan:
        """Generate an execution plan for a complex goal."""
        prompt = f"Goal: {goal}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        response = self.llm.chat(
            messages=[
                Message(role="system", content=PLANNER_SYSTEM),
                Message(role="user", content=prompt),
            ],
            temperature=0.0,
        )

        raw = response.content or "[]"
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
            steps = [PlanStep(**s) for s in data]
        except Exception:
            # Fallback: single step with the original goal
            steps = [PlanStep(step=1, title=goal, description=goal)]

        return Plan(goal=goal, steps=steps)


class PlanExecutor:
    """Executes a Plan step-by-step using a ReActAgent."""

    def __init__(self, agent):  # agent: ReActAgent
        self.agent = agent

    def execute(
        self,
        plan: Plan,
        on_step_start=None,
        on_step_done=None,
    ) -> Plan:
        """
        Execute each step in order.
        Injects results from previous steps into the context of the next.
        """
        context_so_far = []

        for step in plan.steps:
            if on_step_start:
                on_step_start(step)

            # Build task with accumulated context
            task = step.description
            if context_so_far:
                task += "\n\nContext from previous steps:\n" + "\n".join(context_so_far)

            result = self.agent.run(task)
            step.result = result.answer
            step.success = result.success

            context_so_far.append(f"Step {step.step} ({step.title}): {result.answer[:300]}")

            if on_step_done:
                on_step_done(step, result)

            # Reset short-term memory between steps to avoid context pollution
            self.agent.reset()

        return plan
