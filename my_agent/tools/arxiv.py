"""
arXiv RAG Tools
───────────────
Fetch arXiv papers and ingest them into a VectorStore for RAG Q&A.

Uses the free arXiv API (no key required).

Tools registered:
  search_arxiv(query, max_results)      — find papers by keyword/topic
  fetch_arxiv_paper(paper_id)           — ingest a paper into the knowledge base
  list_arxiv_papers()                   — list ingested papers

Usage:
    from tools.arxiv import register_arxiv_tools
    from tools.rag import VectorStore

    store = VectorStore(persist_dir=".agent_memory/arxiv")
    register_arxiv_tools(registry, store)
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from tools.registry import ToolRegistry
from tools.rag import VectorStore


# ── Disk cache (TTL 24 h) ─────────────────────────────────────────────────────

class PaperCache:
    """
    Disk-based cache for arXiv responses (metadata + HTML text).
    Eliminates redundant network calls for papers already fetched today.
    TTL default: 24 hours (papers don't change day-to-day).
    No Redis needed — works everywhere with zero extra dependencies.
    """
    def __init__(self, cache_dir: str = ".arxiv_cache", ttl_hours: int = 24):
        self._dir = cache_dir
        self._ttl = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", key)
        return os.path.join(self._dir, f"{safe}.json")

    def get(self, key: str):
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if time.time() - data["_ts"] > self._ttl:
                os.remove(path)
                return None
            return data["value"]
        except Exception:
            return None

    def set(self, key: str, value) -> None:
        try:
            with open(self._path(key), "w") as f:
                json.dump({"_ts": time.time(), "value": value}, f)
        except Exception:
            pass


# Module-level cache instance — shared across all tool calls in this process
_cache = PaperCache()


# ── arXiv API helpers ─────────────────────────────────────────────────────────

NS = "http://www.w3.org/2005/Atom"


def _fetch_xml(url: str) -> ET.Element:
    req = urllib.request.Request(url, headers={"User-Agent": "arxiv-rag-agent/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return ET.fromstring(resp.read())


def _parse_entries(root: ET.Element) -> List[Dict]:
    entries = []
    for entry in root.findall(f"{{{NS}}}entry"):
        raw_id = entry.findtext(f"{{{NS}}}id", "")
        # Normalise to short ID like "2312.00752" or "cs/0612064"
        paper_id = raw_id.split("/abs/")[-1].strip()

        title = re.sub(r"\s+", " ", entry.findtext(f"{{{NS}}}title", "")).strip()
        summary = re.sub(r"\s+", " ", entry.findtext(f"{{{NS}}}summary", "")).strip()
        published = entry.findtext(f"{{{NS}}}published", "")[:10]

        authors = [
            a.findtext(f"{{{NS}}}name", "")
            for a in entry.findall(f"{{{NS}}}author")
        ]

        # PDF link
        pdf_url = ""
        for link in entry.findall(f"{{{NS}}}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and paper_id:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}"

        entries.append({
            "id":        paper_id,
            "title":     title,
            "authors":   authors,
            "published": published,
            "abstract":  summary,
            "pdf_url":   pdf_url,
            "abs_url":   f"https://arxiv.org/abs/{paper_id}",
        })
    return entries


def search_arxiv_api(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",   # "relevance" | "submittedDate" | "lastUpdatedDate"
    date_from: str = None,        # "YYYYMMDD"
    date_to: str = None,          # "YYYYMMDD"
    category: str = None,         # e.g. "cs.AI", "cs.LG", "cs.CL"
) -> List[Dict]:
    """Search arXiv and return paper metadata."""
    parts = []
    if query.strip():
        parts.append(f"all:{urllib.parse.quote(query)}")
    if category:
        parts.append(f"cat:{category}")
    if date_from or date_to:
        lo = date_from or "00000000"
        hi = date_to   or "99991231"
        parts.append(f"submittedDate:[{lo}0000+TO+{hi}2359]")

    search_query = "+AND+".join(parts) if parts else "all:ai"
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query={search_query}"
        f"&max_results={max_results}"
        f"&sortBy={sort_by}&sortOrder=descending"
    )
    root = _fetch_xml(url)
    return _parse_entries(root)


def fetch_arxiv_paper_api(paper_id: str) -> Optional[Dict]:
    """Fetch a single paper by arXiv ID (cache-first)."""
    clean_id = re.sub(r"v\d+$", "", paper_id.strip())
    cached = _cache.get(f"meta_{clean_id}")
    if cached:
        return cached
    url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
    root = _fetch_xml(url)
    entries = _parse_entries(root)
    result = entries[0] if entries else None
    if result:
        _cache.set(f"meta_{clean_id}", result)
    return result


def _fetch_html_text(paper_id: str) -> Optional[str]:
    """Try to get the HTML version of a paper (cache-first)."""
    clean_id = re.sub(r"v\d+$", "", paper_id.strip())
    cached = _cache.get(f"html_{clean_id}")
    if cached:
        return cached
    url = f"https://arxiv.org/html/{clean_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "arxiv-rag-agent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode(errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        result = text if len(text) > 500 else None
        if result:
            _cache.set(f"html_{clean_id}", result)
        return result
    except Exception:
        return None


def _fetch_pdf_text(pdf_url: str) -> Optional[str]:
    """Download PDF and extract text (needs pypdf)."""
    try:
        import pypdf, io
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "arxiv-rag-agent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if len(text) > 500 else None
    except ImportError:
        return None
    except Exception:
        return None


def build_paper_text(paper: Dict, full_text: Optional[str]) -> str:
    """Combine metadata + full text into a single string for ingestion."""
    authors_str = ", ".join(paper["authors"][:5])
    if len(paper["authors"]) > 5:
        authors_str += f" et al. ({len(paper['authors'])} total)"

    parts = [
        f"Title: {paper['title']}",
        f"Authors: {authors_str}",
        f"Published: {paper['published']}",
        f"arXiv ID: {paper['id']}",
        f"URL: {paper['abs_url']}",
        "",
        "Abstract:",
        paper["abstract"],
    ]

    if full_text:
        # Clean up and limit full text
        cleaned = re.sub(r"\s+", " ", full_text)
        # Trim to ~50k chars to stay within token limits
        parts += ["", "Full Text:", cleaned[:50_000]]
        if len(cleaned) > 50_000:
            parts.append("[... text truncated ...]")

    return "\n".join(parts)


# ── Register as agent tools ───────────────────────────────────────────────────

def register_arxiv_tools(registry: ToolRegistry, store: VectorStore):
    """Register arXiv search + fetch tools onto the given registry."""

    @registry.tool(
        description=(
            "Search arXiv for academic papers by keyword, topic, or author. "
            "THIS IS THE ONLY CORRECT WAY to search arXiv — do NOT use http_request on arxiv.org URLs, "
            "as they return raw HTML that cannot be parsed. "
            "Returns structured titles, authors, abstracts, and IDs. "
            "Use this for ANY question about arXiv papers, recent research, or academic literature. "
            "IMPORTANT: Each result includes an 'ID:' field (e.g. '2312.00752') — "
            "use that exact string when calling fetch_arxiv_paper. "
            "For recent/latest/today queries: set sort_by='submittedDate'. "
            "For date ranges: set date_from and date_to in YYYYMMDD format computed from today's date. "
            "Use category to filter by field: cs.AI (AI), cs.LG (ML), cs.CL (NLP), cs.CV (vision), q-fin (finance)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms, e.g. 'agentic AI' or 'retrieval augmented generation'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return (default 5, max 20)",
                    "default": 5,
                },
                "sort_by": {
                    "type": "string",
                    "description": "'relevance' (default) or 'submittedDate' (use for latest/recent/today queries)",
                    "enum": ["relevance", "submittedDate", "lastUpdatedDate"],
                    "default": "relevance",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date filter YYYYMMDD, e.g. '20260220'",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date filter YYYYMMDD, e.g. '20260222'",
                },
                "category": {
                    "type": "string",
                    "description": "arXiv category filter, e.g. 'cs.AI', 'cs.LG', 'cs.CL'",
                },
            },
            "required": ["query"],
        },
        tags=["arxiv", "rag", "knowledge"],
    )
    def search_arxiv(
        query: str,
        max_results: int = 5,
        sort_by: str = "relevance",
        date_from: str = None,
        date_to: str = None,
        category: str = None,
    ) -> str:
        max_results = min(max_results, 20)
        papers = search_arxiv_api(query, max_results, sort_by, date_from, date_to, category)
        if not papers:
            return "No papers found for that query."
        lines = []
        for p in papers:
            authors_str = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors_str += " et al."
            lines.append(
                f"ID: {p['id']}\n"
                f"Title: {p['title']}\n"
                f"Authors: {authors_str}\n"
                f"Published: {p['published']}\n"
                f"Abstract: {p['abstract'][:300]}...\n"
                f"URL: {p['abs_url']}"
            )
        return f"Found {len(papers)} papers:\n\n" + "\n\n---\n\n".join(lines)

    @registry.tool(
        description=(
            "Fetch an arXiv paper by its ID and ingest it into the knowledge base for Q&A. "
            "This downloads the paper text (abstract + full content when available) "
            "and stores it so you can answer questions about it. "
            "IMPORTANT: paper_id must be the ACTUAL numeric arXiv ID from search results "
            "(e.g. '2312.00752', '2005.11401', 'cs/0612064'). "
            "Do NOT use placeholder text like 'ID of the first paper' — copy the exact ID string "
            "from the 'ID:' field in search_arxiv results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "arXiv paper ID, e.g. '2005.11401'",
                },
                "include_full_text": {
                    "type": "boolean",
                    "description": "Also download full paper text beyond the abstract (default true)",
                    "default": True,
                },
            },
            "required": ["paper_id"],
        },
        tags=["arxiv", "rag", "knowledge"],
    )
    def fetch_arxiv_paper(paper_id: str, include_full_text: bool = True) -> str:
        # Validate that paper_id looks like a real arXiv ID (not a placeholder)
        if not re.match(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7}(v\d+)?)$", paper_id.strip()):
            return (
                f"Invalid arXiv ID: '{paper_id}'. "
                "Please use the exact numeric ID from search results, e.g. '2312.00752' or '2005.11401'. "
                "Run search_arxiv first and copy the 'ID:' field from the output."
            )

        paper = fetch_arxiv_paper_api(paper_id)
        if not paper:
            return f"Could not find paper with ID: {paper_id}"

        full_text = None
        source = "abstract only"
        if include_full_text:
            # Try HTML first (cleaner), then PDF
            full_text = _fetch_html_text(paper["id"])
            if full_text:
                source = "HTML full text"
            else:
                full_text = _fetch_pdf_text(paper["pdf_url"])
                if full_text:
                    source = "PDF full text"

        text = build_paper_text(paper, full_text)
        doc_name = f"arxiv:{paper['id']} - {paper['title'][:60]}"
        n_chunks = store.ingest_text(text, doc_name, metadata={
            "arxiv_id": paper["id"],
            "title": paper["title"],
            "published": paper["published"],
            "source": source,
        })

        return (
            f"Ingested '{paper['title']}' ({paper['id']})\n"
            f"Source: {source}\n"
            f"Chunks: {n_chunks}\n"
            f"URL: {paper['abs_url']}\n"
            f"The paper is now in the knowledge base. Use search_knowledge_base to query it."
        )

    @registry.tool(
        description=(
            "Fetch multiple arXiv papers in parallel and ingest them all at once. "
            "Use this instead of calling fetch_arxiv_paper one by one when you have 2+ paper IDs. "
            "Provide the exact ID strings from search_arxiv results (e.g. ['2312.00752', '2005.11401'])."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of arXiv IDs to fetch and ingest (max 10)",
                },
                "include_full_text": {
                    "type": "boolean",
                    "description": "Download full paper text (default true)",
                    "default": True,
                },
            },
            "required": ["paper_ids"],
        },
        tags=["arxiv", "rag", "knowledge"],
    )
    def fetch_arxiv_papers_batch(paper_ids: list, include_full_text: bool = True) -> str:
        if not paper_ids:
            return "No paper IDs provided."

        valid_ids, invalid_ids = [], []
        for pid in paper_ids[:10]:
            if re.match(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7}(v\d+)?)$", str(pid).strip()):
                valid_ids.append(str(pid).strip())
            else:
                invalid_ids.append(pid)

        if not valid_ids:
            return (
                "No valid arXiv IDs in the list. "
                "Use search_arxiv first and copy the exact 'ID:' values."
            )

        def _ingest_one(paper_id: str) -> str:
            paper = fetch_arxiv_paper_api(paper_id)
            if not paper:
                return f"  ✗ {paper_id}: not found"
            full_text, source = None, "abstract only"
            if include_full_text:
                full_text = _fetch_html_text(paper["id"])
                source = "HTML" if full_text else "abstract only"
                if not full_text:
                    full_text = _fetch_pdf_text(paper["pdf_url"])
                    source = "PDF" if full_text else "abstract only"
            text = build_paper_text(paper, full_text)
            doc_name = f"arxiv:{paper['id']} - {paper['title'][:60]}"
            n = store.ingest_text(text, doc_name, {
                "arxiv_id": paper["id"], "title": paper["title"],
                "published": paper["published"], "source": source,
            })
            return f"  ✓ {paper['id']}: {paper['title'][:50]} ({n} chunks, {source})"

        lines = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_ingest_one, pid): pid for pid in valid_ids}
            for future in as_completed(futures):
                lines.append(future.result())

        msg = f"Ingested {len(valid_ids)} papers in parallel:\n" + "\n".join(sorted(lines))
        if invalid_ids:
            msg += f"\nSkipped invalid IDs: {invalid_ids}"
        msg += "\nUse search_knowledge_base to query them."
        return msg

    @registry.tool(
        description="List all arXiv papers currently in the knowledge base.",
        input_schema={"type": "object", "properties": {}, "required": []},
        tags=["arxiv", "rag", "knowledge"],
    )
    def list_arxiv_papers() -> str:
        docs = store.list_documents()
        arxiv_docs = [d for d in docs if d["doc_name"].startswith("arxiv:")]
        if not arxiv_docs:
            return "No arXiv papers in the knowledge base yet. Use fetch_arxiv_paper to add some."
        lines = [f"  {d['doc_name']}  ({d['chunks']} chunks)" for d in arxiv_docs]
        return f"{len(arxiv_docs)} papers indexed:\n" + "\n".join(lines)
