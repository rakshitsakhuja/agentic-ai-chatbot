# AI Chatbot — Architecture

> Last updated: 2026-02-22

---

## Full System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Streamlit UI  (app.py)                          │
│                                                                          │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │     Chat window      │  │                Sidebar                  │  │
│  │                      │  │  Provider · Model · API key             │  │
│  │  user message        │  │  Persona · Custom system prompt         │  │
│  │  ⏳ Thinking… (poll) │  │  Max iterations slider (3–25, dflt 15)  │  │
│  │  ⏹ Stop button       │  │  Tools toggle · arXiv RAG toggle        │  │
│  │  agent answer        │  │  arXiv search · Add by ID · Paper list  │  │
│  │  tool call expander  │  │  Ingest progress (⏳⚙️✅❌)             │  │
│  │  trace badge (cost,  │  │  Dynamic tools list · Delete tool        │  │
│  │    tokens, latency)  │  │  Langfuse toggle · Session stats        │  │
│  └──────────┬───────────┘  └─────────────────────────────────────────┘  │
└─────────────┼────────────────────────────────────────────────────────────┘
              │ user submits prompt
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Background State Machine  (app.py)                    │
│                                                                          │
│  @st.cache_resource — objects created once, survive all Streamlit reruns │
│                                                                          │
│   _run_state  dict  ←→  threading.Lock  (_run_state_lock)               │
│   status: "idle" → "running" → "done"|"stopped"|"error" → "idle"        │
│                                                                          │
│   Streamlit main thread:                                                 │
│     • if prompt: start daemon thread, set status="running", st.rerun()  │
│     • if status=="running": render spinner + Stop button,               │
│         time.sleep(0.5) → st.rerun()   (polls every 0.5 s)             │
│     • if status=="done": append answer to session_state, st.rerun()     │
│                                                                          │
│   Background thread (_run_agent_worker):                                 │
│     tracer.run(agent, prompt) → sets status="done"|"error" + result     │
│                                                                          │
│   Ingest queue (_ingest_jobs) also @st.cache_resource                   │
│     sidebar rerun guard: only calls st.rerun() when agent is idle       │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ agent.run(prompt)
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Langfuse v3 Tracer  (tracer.py)                       │
│   wraps agent.run() inside start_as_current_span("agent_turn")          │
│                                                                          │
│   Callbacks connected to agent:                                          │
│     on_thought   → start_observation(as_type="generation")              │
│                    completion_start_time = actual LLM call start        │
│     on_tool_call → start_observation(as_type="tool")                    │
│     on_tool_result → obs.update(output, level) → obs.end()             │
│                                                                          │
│   All iterations get a generation span (even tool-only iterations)      │
│   flush() runs in a daemon thread so it never blocks the response       │
└──────────────┬───────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop  (react.py)                          │
│                                                                          │
│   System prompt (injected at init):                                      │
│     • Today's date  → "2026-02-22" / "20260222" (arXiv filter format)  │
│     • Tool guidance → prefer specialized tools, never http_request on   │
│       arxiv.org, proactively create tools when capability is missing    │
│     • Naming rules  → tool names must be generic (scrape_url, not       │
│       crawl_langchain_docs); check list_dynamic_tools before creating   │
│     • Persona       → passed in as extra_instructions                   │
│                                                                          │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────────────────┐   │
│   │  THINK   │────▶│   ACT    │────▶│         OBSERVE              │   │
│   │  LLM call│     │ tool call│     │ result truncated to 3000 chars│   │
│   └──────────┘     └──────────┘     └──────────────┬───────────────┘   │
│        ▲                                            │                   │
│        └────────────────────────────────────────────┘                   │
│                                                                          │
│   ShortTermMemory (sliding window, max_messages=50)                      │
│     _trim(): advances cut to nearest user message — never splits an     │
│     assistant+tool_results group (prevents orphaned tool messages)      │
│                                                                          │
│   Deduplication: (name, args) hash — max 5 tool calls per iteration    │
│   Retry guard: identical failing call ≥ 2× → inject abort hint         │
│   Max iterations: configurable via sidebar (default 15); when hit,     │
│     memory.add_assistant() called to close any open tool-call cycle    │
│   Stop: threading.Event checked at start of every iteration            │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ ToolRouter selects ≤ max_tools tools per iteration
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│               ToolRouter  (router.py)  +  ToolRegistry  (registry.py)   │
│                                                                          │
│   ToolRouter — keyword-based group selection (re-evaluated every iter)  │
│                                                                          │
│   Always-on:  python_repl · list_dynamic_tools                          │
│                                                                          │
│   Groups (activated by keyword match on the user query):                │
│     "file"        → read_file · write_file · search_files               │
│     "shell"       → run_shell                                            │
│     "web"         → http_request                                         │
│     "arxiv"       → search_arxiv · fetch_arxiv_paper ·                  │
│                     fetch_arxiv_papers_batch · list_arxiv_papers ·      │
│                     search_knowledge_base  ← always paired with arxiv   │
│     "rag"         → search_knowledge_base · ingest_document ·           │
│                     list_documents                                       │
│     "dynamic_meta"→ create_tool · list_dynamic_tools ·                  │
│                     delete_dynamic_tool                                  │
│                                                                          │
│   Cap: max_tools=10; trim order (lowest priority first):                │
│     dynamic_meta → web → rag → shell → file → arxiv                    │
│                                                                          │
│   Custom dynamic tools appended last (always visible once created)      │
└──────────────┬───────────────────────────────────────────────────────────┘
               │
       ┌───────┴────────────────────────────────────┐
       ▼                                             ▼
┌────────────────────────────┐      ┌───────────────────────────────────┐
│       arXiv Tools          │      │     Dynamic Tool Synthesis        │
│                            │      │          (dynamic.py)             │
│  search_arxiv              │      │                                   │
│    └─ arXiv REST API       │      │  create_tool(name, desc, code)    │
│       date-aware filters   │      │    1. exec() function definition  │
│                            │      │    2. register in live registry   │
│  fetch_arxiv_paper         │      │    3. save to .agent_tools/       │
│    └─ PaperCache (24h TTL) │      │       tools.json                  │
│       HTML → PDF fallback  │      │    4. auto-loads on next session  │
│                            │      │                                   │
│  fetch_arxiv_papers_batch  │      │  Naming rules enforced at prompt  │
│    └─ ThreadPoolExecutor   │      │  and tool description level:      │
│       max_workers=5        │      │  name = WHAT it does generically  │
│       all IDs in parallel  │      │  (scrape_url ✓, crawl_langchain ✗)│
│                            │      │                                   │
│  list_arxiv_papers         │      │  list_dynamic_tools always-on     │
└────────────┬───────────────┘      │  → agent checks before creating  │
             │                      └───────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline  (rag.py)                             │
│                                                                          │
│  VectorStore(.arxiv_store/)                                              │
│    ├── SemanticChunker  (primary: semantic-text-splitter Rust lib)       │
│    │     target_size=600, overlap_sentences=2, min_chunk_size=80        │
│    │     MarkdownSplitter if headers detected, else TextSplitter        │
│    │     Fallback: pure Python regex chunker                            │
│    │                                                                     │
│    ├── ingest_text(text, doc_name) → chunk → embed → store (pickle)    │
│    │                                                                     │
│    └── search(query, top_k=5)                                           │
│          embed(query) → cosine similarity → deduplicate by doc_id       │
│                                                                          │
│  BM25Retriever  (fallback when no embedder)                             │
│    k1=1.5, b=0.75 — better term saturation than TF-IDF                 │
│    scores at query time, dedupes by doc_id                              │
│                                                                          │
│  Embedding backends (auto-detected, best available wins):               │
│    Priority 1 ── OpenAIEmbedder  (text-embedding-3-small, paid)        │
│    Priority 2 ── SentenceTransformerEmbedder  (all-MiniLM-L6-v2, free) │
│    Priority 3 ── BM25Retriever  (keyword fallback, zero deps)          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: "Latest vLLM deployment papers?"

```
User  →  "Find the latest papers on vLLM deployment and summarise them"
           │
           ▼
      ToolRouter.select(query)
        "papers" → arxiv group + search_knowledge_base (always paired)
        Selected: search_arxiv, fetch_arxiv_papers_batch,
                  search_knowledge_base, python_repl, list_dynamic_tools
           │
     [iter 1] THINK: search arXiv
           │   ACT:  search_arxiv(query="vLLM deployment", max_results=5)
           │   OBSERVE: "Found 5 papers: ID: 2602.xxxxx ..."
           │
     [iter 2] THINK: fetch paper content
           │   ACT:  fetch_arxiv_papers_batch(paper_ids=[...])
           │         ┌── Thread 1: HTML → chunk → embed → store ──┐
           │         └── Thread 2: HTML → chunk → embed → store ──┘ parallel
           │   OBSERVE: "Ingested 2 papers. Use search_knowledge_base."
           │
     [iter 3] THINK: query the knowledge base
           │   ACT:  search_knowledge_base(query="vLLM deployment", top_k=5)
           │         embed(query) → cosine similarity → dedupe by doc_id
           │   OBSERVE: "[1] ThunderAgent (score: 0.85): vLLM is used as..."
           │
     [final] ANSWER: "Based on recent papers: ..."  (cites paper ID + title)
```

---

## Background Thread Lifecycle

```
Streamlit script runs top-to-bottom on every rerun:

  Run N  (user submits):
    state handler → status=idle → skip
    [sidebar, header, history rendered]
    if prompt:
      append user message to session_state.messages
      start daemon thread → _run_agent_worker(agent, prompt)
      _run_state["status"] = "running"
      st.rerun()

  Run N+1 … N+K  (polling, every 0.5 s):
    state handler → status=running → render spinner + Stop button
    time.sleep(0.5) → st.rerun()

  Background thread (after ~Xs):
    tracer.run() completes
    _run_state["status"] = "done"
    _run_state["result"] = AgentResult(...)

  Run N+K+1  (picks up done):
    state handler → status=done
      reset _run_state to idle
      append assistant message to session_state.messages
      st.rerun()

  Run N+K+2  (clean idle render):
    status=idle → skip state handler
    render full chat history including new answer ✓
```

---

## Component Map

| File | Role |
|---|---|
| `chatbot_app/app.py` | Streamlit UI, `@st.cache_resource` state, background polling, sidebar |
| `my_agent/agent/react.py` | ReAct loop, memory, dedup, stop_event, max_iterations |
| `my_agent/llm/openai_llm.py` | OpenAI-compatible LLM (Groq, Ollama, DeepSeek); Groq malformed-tool recovery |
| `my_agent/llm/anthropic_llm.py` | Anthropic Claude adapter |
| `my_agent/memory/short_term.py` | Sliding window; trim advances to user boundary (no orphaned tool msgs) |
| `my_agent/tools/registry.py` | Tool registration + `execute(inputs or {})` |
| `my_agent/tools/router.py` | Keyword-based group selection; `list_dynamic_tools` always-on |
| `my_agent/tools/rag.py` | VectorStore, BM25, SemanticChunker, SentenceTransformer / OpenAI embedder |
| `my_agent/tools/arxiv.py` | PaperCache (24h TTL), search, fetch single + batch, list |
| `my_agent/tools/builtin.py` | Shell, file, HTTP, Python REPL |
| `my_agent/tools/dynamic.py` | Dynamic tool synthesis; `create_tool`, `list_dynamic_tools`, `delete_dynamic_tool` |
| `my_agent/observability/tracer.py` | Langfuse v3 SPAN/GENERATION/TOOL; daemon-thread flush |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `@st.cache_resource` for `_run_state` | Plain module-level `x = {}` re-runs on every Streamlit rerun, resetting state and disconnecting background threads. `@st.cache_resource` creates the object once and returns the same instance forever. |
| 0.5 s polling via `time.sleep` + `st.rerun()` | Streamlit is synchronous; true async requires unsupported `asyncio` integration. Polling is the standard pattern. |
| Sidebar ingest rerun guard | Sidebar `st.rerun()` (for ingest progress) ran before the state handler, preventing the done state from ever being seen. Guard skips the rerun when agent status ≠ idle. |
| `search_knowledge_base` in arxiv group | `fetch_arxiv_papers_batch` output says "use search_knowledge_base". Without it in the arxiv tool group, the agent never had it available — causing endless search→ingest loops hitting max iterations. |
| `list_dynamic_tools` always-on | Agent must check existing tools before creating new ones. Making it always available ensures the check happens even when `dynamic_meta` keywords don't appear in the query. |
| Memory trim to user boundary | Naive `[-N:]` trim can slice an `assistant_with_tool_calls` + `tool_result` pair, leaving an orphaned tool message that causes API 400 errors on every subsequent call. |
| `memory.add_assistant()` on max iterations | Without this, memory ends with bare `tool_result` messages. The next turn's `add_user()` creates an invalid `[..., tool_result, user]` sequence rejected by the API. |
| Generic tool naming enforced at two layers | System prompt rules + `create_tool` parameter description both specify "describe WHAT, not WHERE". Two enforcement points reduce task-specific name generation (e.g. `crawl_langchain_docs` → `scrape_url`). |
| `flush()` in daemon thread | Langfuse flush was blocking the background worker thread, delaying `_run_state["status"] = "done"` and slowing down UI response. |

---

## Current Limitations

| # | Area | Problem |
|---|---|---|
| 1 | **Vector store** | Pickle file — not safe for concurrent multi-user writes |
| 2 | **No re-ranking** | Embedding cosine score is final — no cross-encoder second pass |
| 3 | **Multi-user** | All users share one VectorStore — papers not isolated per user |
| 4 | **arXiv rate limits** | No retry/backoff — throttle from arXiv = immediate error |
| 5 | **Version IDs** | `2602.17665v1` and `v2` stored as separate docs (same paper twice) |
| 6 | **Small models** | llama-3.1-8b picks wrong tools; use 70B+ for reliable tool selection |
| 7 | **Dynamic tool sandbox** | `exec()` runs in same process — no isolation; unsafe for untrusted code |
| 8 | **No streaming** | Full response appears at once; no token-by-token streaming in UI |
| 9 | **Session persistence** | Chat history lost on browser refresh (stored in `st.session_state` only) |

---

## Production Upgrade Path

```
Current (free, local)               Production
────────────────────────────────────────────────────────────────────────
Embedding   all-MiniLM-L6-v2        text-embedding-3-large (OpenAI)
Store       Pickle file             ChromaDB / pgvector / Pinecone
Search      Dense cosine only       Hybrid: BM25 + dense + cross-encoder rerank
Chunking    SemanticChunker (600)   Docling / Unstructured for PDF/table parsing
Queue       threading.Thread        Celery + Redis (multi-process safe)
Job state   @st.cache_resource      Redis (survives restarts, multi-user)
Isolation   Single VectorStore      Namespace per user/session
Resilience  No retry                tenacity exponential backoff on arXiv API
Tool exec   exec() same process     subprocess + RestrictedPython sandbox
History     session_state only      SQLite / Postgres per session_id
Streaming   None                    st.write_stream() + async generator
```
