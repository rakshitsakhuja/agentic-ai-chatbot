"""
Tool Router — query-time tool selection
────────────────────────────────────────
Keeps the tool count under model limits (~10) by selecting only the groups
relevant to each user query, instead of dumping every registered tool into
every API call.

Why this matters:
  Groq's llama-3.3-70b-versatile (and similar models) start generating
  malformed tool calls once the schema payload exceeds ~10-12 tools.
  The chatbot registers 17+ tools when all features are enabled, which
  reliably triggers `failed_generation` 400 errors.

Routing strategy:
  1. Core tools are always included (python_repl).
  2. Keyword patterns activate tool groups.
  3. If nothing matches beyond core, a safe default set is used.
  4. Custom / dynamic tools are appended last (in creation order).
  5. Total is capped at `max_tools` (default 10).
"""

import re
from typing import Collection, Set

MAX_TOOLS = 10

# ── Tool groups ───────────────────────────────────────────────────────────────

_ALWAYS: Set[str] = {"python_repl", "list_dynamic_tools"}

_GROUPS: dict[str, Set[str]] = {
    "file":         {"read_file", "write_file", "search_files"},
    "shell":        {"run_shell"},
    "web":          {"http_request"},
    # search_knowledge_base is included in arxiv group because fetch_arxiv_papers_batch
    # ingests into the knowledge base and the agent must be able to query it immediately
    # after fetching — both tools go together regardless of which keyword triggered routing.
    "arxiv":        {"search_arxiv", "fetch_arxiv_paper", "fetch_arxiv_papers_batch", "list_arxiv_papers", "search_knowledge_base"},
    "rag":          {"search_knowledge_base", "ingest_document", "list_documents"},
    "dynamic_meta": {"create_tool", "list_dynamic_tools", "delete_dynamic_tool"},
}

# When we must trim, remove these groups first (lowest → highest priority)
_TRIM_ORDER = ["dynamic_meta", "web", "rag", "shell", "file", "arxiv"]

# Default set when no keyword group matched (avoids exposing nothing)
_DEFAULT: Set[str] = _GROUPS["file"] | _GROUPS["shell"]

# ── Keyword patterns ──────────────────────────────────────────────────────────

_PATTERNS: dict[str, re.Pattern] = {
    "file": re.compile(
        r"\b(file|read|write|open|load|save|export|document|csv|json|txt|pdf|"
        r"directory|folder|path|content|import|output)\b",
        re.I,
    ),
    "shell": re.compile(
        r"\b(run|execute|terminal|shell|command|bash|script|install|pip|npm|"
        r"git|build|compile|process|stdout|stderr)\b",
        re.I,
    ),
    "web": re.compile(
        r"(https?://|\b(fetch|url|endpoint|request|website|webpage|download|scrape|crawl)\b)",
        re.I,
    ),
    "arxiv": re.compile(
        r"\b(arxiv|paper|papers|research|publication|preprint|abstract|doi|"
        r"citation|citations|author|journal|study|studies|findings|literature)\b",
        re.I,
    ),
    "rag": re.compile(
        r"\b(search|knowledge|indexed|stored|ingested|retrieve|"
        r"what.{0,10}know|from.{0,15}paper|based.{0,15}paper|in.{0,10}database)\b",
        re.I,
    ),
    "dynamic_meta": re.compile(
        r"\b(create.{0,10}tool|new\s+tool|make.{0,10}tool|add.{0,10}tool|"
        r"save.{0,10}(function|tool)|reusable|list.{0,10}tool|delete.{0,10}tool)\b",
        re.I,
    ),
}

# All tools that belong to a predefined group (used to identify custom tools)
_KNOWN: Set[str] = _ALWAYS | set().union(*_GROUPS.values())


# ── Router ────────────────────────────────────────────────────────────────────

class ToolRouter:
    """
    Selects a subset of available tools relevant to a user query.

    Usage:
        router = ToolRouter(max_tools=10)
        names  = router.select("find recent papers on transformers", all_tool_names)
        schema = registry.as_schema(names=names)
    """

    def __init__(self, max_tools: int = MAX_TOOLS):
        self.max_tools = max_tools

    def select(self, query: str, available: Collection[str]) -> Set[str]:
        """
        Return a set of tool names relevant to `query`, drawn from `available`.
        Result size never exceeds self.max_tools.
        """
        avail = set(available)
        selected: Set[str] = set()

        # 1. Always-on core tools
        selected |= _ALWAYS & avail

        # 2. Activate groups whose keyword pattern matches the query
        for group, pattern in _PATTERNS.items():
            if pattern.search(query):
                selected |= _GROUPS.get(group, set()) & avail

        # 3. Nothing matched beyond core → use safe defaults
        if selected <= (_ALWAYS & avail):
            selected |= _DEFAULT & avail

        # 4. Trim to max_tools (remove lowest-priority groups first)
        if len(selected) > self.max_tools:
            for group in _TRIM_ORDER:
                if len(selected) <= self.max_tools:
                    break
                selected -= _GROUPS.get(group, set()) & selected

        # 5. Fill remaining slots with custom (dynamic) tools not in any group
        custom = avail - _KNOWN
        for name in sorted(custom):              # deterministic order
            if len(selected) >= self.max_tools:
                break
            selected.add(name)

        return selected

    def explain(self, query: str, available: Collection[str]) -> str:
        """Human-readable explanation of which groups were activated."""
        avail = set(available)
        matched = [g for g, pat in _PATTERNS.items() if pat.search(query)]
        selected = self.select(query, avail)
        return (
            f"Query: {query!r}\n"
            f"Matched groups: {matched or ['(none — using defaults)']}\n"
            f"Selected tools ({len(selected)}): {sorted(selected)}"
        )
