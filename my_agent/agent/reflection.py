"""
Reflection  (Critic / Self-Evaluation Pattern)
────────────────────────────────────────────────
After the ReAct agent produces an answer, the Reflector:
  1. Evaluates whether the answer is complete, correct, and safe
  2. If not — returns critique + suggested improvements
  3. The agent can then retry with the critique as guidance

Pattern: Generate → Critique → Refine → repeat up to max_retries

Use when:
  - High-stakes decisions (trades, financial recommendations)
  - Quality matters more than speed
  - You want explainable, self-audited outputs

Usage:
    reflector = Reflector(llm)
    refined = reflector.reflect_and_refine(agent, task, initial_answer)
"""

from dataclasses import dataclass
from typing import Optional

from llm.base import BaseLLM, Message
from agent.react import ReActAgent, AgentResult


CRITIC_SYSTEM = """You are a strict quality evaluator for an AI agent's output.

Given: (1) the original task, (2) the agent's answer.

Evaluate on:
1. COMPLETENESS — does it fully address the task?
2. ACCURACY     — is it factually correct / consistent?
3. SAFETY       — does it follow any stated rules / constraints?
4. CLARITY      — is it actionable and unambiguous?

Respond in this exact JSON format:
{
  "passed": true | false,
  "score": 0-10,
  "issues": ["issue1", "issue2"],
  "suggestions": "What the agent should do differently to improve the answer"
}

Be strict. Only pass answers that are complete, accurate, and safe."""


@dataclass
class ReflectionResult:
    passed: bool
    score: int
    issues: list
    suggestions: str
    final_answer: str
    attempts: int


class Reflector:
    def __init__(self, llm: BaseLLM, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def critique(self, task: str, answer: str) -> dict:
        """Run the critic LLM pass. Returns parsed evaluation dict."""
        import json
        response = self.llm.chat(
            messages=[
                Message(role="system", content=CRITIC_SYSTEM),
                Message(role="user", content=f"Task:\n{task}\n\nAgent Answer:\n{answer}"),
            ],
            temperature=0.0,
        )
        raw = (response.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except Exception:
            return {"passed": True, "score": 7, "issues": [], "suggestions": ""}

    def reflect_and_refine(
        self,
        agent: ReActAgent,
        task: str,
        initial_result: AgentResult,
        verbose: bool = True,
    ) -> ReflectionResult:
        """
        Critique → if fail → retry agent with critique feedback → repeat.
        Returns the best answer found.
        """
        current_answer = initial_result.answer
        attempts = 1

        for attempt in range(self.max_retries + 1):
            evaluation = self.critique(task, current_answer)
            passed = evaluation.get("passed", True)
            score = evaluation.get("score", 7)
            issues = evaluation.get("issues", [])
            suggestions = evaluation.get("suggestions", "")

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"\n[Reflection] {status} score={score}/10 issues={issues}")

            if passed or attempt == self.max_retries:
                return ReflectionResult(
                    passed=passed,
                    score=score,
                    issues=issues,
                    suggestions=suggestions,
                    final_answer=current_answer,
                    attempts=attempts,
                )

            # Retry: inject critique into the task
            if verbose:
                print(f"[Reflection] Retrying with feedback: {suggestions}")

            refined_task = (
                f"{task}\n\n"
                f"[IMPORTANT] Your previous attempt had issues:\n"
                f"Issues: {', '.join(issues)}\n"
                f"Please improve by: {suggestions}"
            )
            agent.reset()
            retry_result = agent.run(refined_task)
            current_answer = retry_result.answer
            attempts += 1

        return ReflectionResult(
            passed=False,
            score=0,
            issues=["Max retries reached"],
            suggestions="",
            final_answer=current_answer,
            attempts=attempts,
        )
