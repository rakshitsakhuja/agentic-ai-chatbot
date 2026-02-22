"""
Query History — Few-Shot SQL Cache
────────────────────────────────────
Stores successful NL→SQL pairs.
On each new question, retrieves the most similar past queries
and injects them as few-shot examples into the agent's context.

Why this matters:
  - Your warehouse has naming conventions the LLM doesn't know
  - Past correct queries teach the agent YOUR SQL patterns
  - Avoids repeating the same schema-search steps for common questions
  - Gets dramatically more accurate over time (self-improving)

Usage:
    history = QueryHistory(".agent_memory/query_history.jsonl")
    register_query_history_tools(registry, history)

    # Agent calls:
    #   get_similar_queries("top customers by revenue")
    #     → returns 3 past SQL queries for similar questions
    #   save_successful_query(question, sql, result_preview)
    #     → saves this pair for future reference
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class QueryRecord:
    id: str
    timestamp: float
    question: str
    sql: str
    result_preview: str          # first few rows as string
    tables_used: List[str]
    row_count: int
    feedback: str = "good"       # "good" | "bad" | "fixed"


class QueryHistory:
    def __init__(self, path: str = ".agent_memory/query_history.jsonl"):
        self._path = path
        self._records: List[QueryRecord] = []
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    def save(
        self,
        question: str,
        sql: str,
        result_preview: str = "",
        tables_used: List[str] = None,
        row_count: int = 0,
    ) -> QueryRecord:
        record = QueryRecord(
            id=f"q_{int(time.time() * 1000)}",
            timestamp=time.time(),
            question=question,
            sql=sql,
            result_preview=result_preview[:500],
            tables_used=tables_used or self._extract_tables(sql),
            row_count=row_count,
        )
        self._records.append(record)
        self._append(record)
        return record

    def search(self, question: str, top_k: int = 3) -> List[Tuple[float, QueryRecord]]:
        """Keyword similarity search over past questions."""
        keywords = set(self._tokenize(question))
        scored = []
        for r in self._records:
            past_kw = set(self._tokenize(r.question))
            if not past_kw:
                continue
            overlap = len(keywords & past_kw)
            score = overlap / max(len(keywords | past_kw), 1)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return scored[:top_k]

    def format_as_examples(self, records: List[Tuple[float, QueryRecord]]) -> str:
        """Format past queries as few-shot examples for the LLM."""
        if not records:
            return "No similar past queries found."
        parts = ["Similar past queries (use as reference for SQL patterns):"]
        for i, (score, r) in enumerate(records, 1):
            parts.append(
                f"\n[Example {i}] (similarity={score:.0%})\n"
                f"Question: {r.question}\n"
                f"SQL:\n{r.sql}\n"
                f"Tables used: {', '.join(r.tables_used)}"
            )
        return "\n".join(parts)

    def get_recent(self, n: int = 10) -> List[QueryRecord]:
        return self._records[-n:]

    def mark_feedback(self, query_id: str, feedback: str):
        for r in self._records:
            if r.id == query_id:
                r.feedback = feedback
        self._rewrite()

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table references from SQL using regex."""
        pattern = r'(?:FROM|JOIN)\s+([\w.]+)'
        return list(set(re.findall(pattern, sql, re.IGNORECASE)))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    for line in f:
                        if line.strip():
                            self._records.append(QueryRecord(**json.loads(line)))
            except Exception:
                pass

    def _append(self, record: QueryRecord):
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def _rewrite(self):
        with open(self._path, "w") as f:
            for r in self._records:
                f.write(json.dumps(asdict(r)) + "\n")


def register_query_history_tools(registry, history: QueryHistory):

    from tools.registry import ToolRegistry

    @registry.tool(
        description=(
            "Search past successful SQL queries similar to the current question. "
            "Call this BEFORE writing SQL — past examples show the correct table names, "
            "join patterns, and column conventions for your warehouse."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The current question"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["question"],
        },
        tags=["sql", "history"],
    )
    def get_similar_queries(question: str, top_k: int = 3) -> str:
        results = history.search(question, top_k)
        return history.format_as_examples(results)

    @registry.tool(
        description=(
            "Save a successful NL→SQL query pair to history for future reference. "
            "Call this after executing a query that returned correct results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "sql":      {"type": "string"},
                "result_preview": {"type": "string", "default": ""},
                "row_count": {"type": "integer", "default": 0},
            },
            "required": ["question", "sql"],
        },
        tags=["sql", "history"],
    )
    def save_successful_query(
        question: str, sql: str, result_preview: str = "", row_count: int = 0
    ) -> str:
        record = history.save(question, sql, result_preview, row_count=row_count)
        return f"Saved query {record.id} to history."
