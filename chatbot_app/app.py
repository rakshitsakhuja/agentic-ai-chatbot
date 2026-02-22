"""
Chatbot App — Streamlit UI with Langfuse Tracing
──────────────────────────────────────────────────
Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Cloud → done
"""

import os
import sys
import time
import threading
import streamlit as st
import streamlit.components.v1 as components

# ── Import agent framework ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "my_agent"))

# ── Background ingest queue ───────────────────────────────────────────────────
# IMPORTANT: use @st.cache_resource so the dict is NOT re-created on every
# Streamlit rerun. Plain module-level assignments (x = {}) ARE re-executed on
# every rerun, which would reset the shared state and disconnect background
# threads from the main script. @st.cache_resource creates the object once and
# returns the same instance on every subsequent call.
# Keys: paper_id  →  {"status": "queued|processing|done|error", "title": str, "error": str}

@st.cache_resource(show_spinner=False)
def _get_ingest_state():
    return {}, threading.Lock()

_ingest_jobs, _ingest_lock = _get_ingest_state()


def _bg_ingest_worker(paper_id: str, store) -> None:
    """Run in a background thread. Fetches + ingests one paper."""
    from tools.arxiv import fetch_arxiv_paper_api, _fetch_html_text, _fetch_pdf_text, build_paper_text
    with _ingest_lock:
        _ingest_jobs[paper_id]["status"] = "processing"
    try:
        paper = fetch_arxiv_paper_api(paper_id)
        if not paper:
            with _ingest_lock:
                _ingest_jobs[paper_id].update({"status": "error", "error": "Paper not found"})
            return
        full_text = _fetch_html_text(paper["id"]) or _fetch_pdf_text(paper["pdf_url"])
        text = build_paper_text(paper, full_text)
        doc_name = f"arxiv:{paper['id']} - {paper['title'][:60]}"
        store.ingest_text(text, doc_name, {"arxiv_id": paper["id"], "title": paper["title"]})
        with _ingest_lock:
            _ingest_jobs[paper_id].update({"status": "done", "title": paper["title"][:50]})
    except Exception as e:
        import traceback
        print(f"\n[ingest worker ERROR] {paper_id}\n{traceback.format_exc()}", flush=True)
        with _ingest_lock:
            _ingest_jobs[paper_id].update({"status": "error", "error": str(e)})


def queue_ingest(paper_id: str, title: str, store) -> None:
    """Add a paper to the background ingest queue and start a worker thread."""
    with _ingest_lock:
        if paper_id in _ingest_jobs and _ingest_jobs[paper_id]["status"] in ("queued", "processing"):
            return  # already in flight
        _ingest_jobs[paper_id] = {"status": "queued", "title": title, "error": ""}
    t = threading.Thread(target=_bg_ingest_worker, args=(paper_id, store), daemon=True)
    t.start()


# ── Background agent run state ────────────────────────────────────────────────
# Same pattern as _ingest_state: use @st.cache_resource so the dict and lock
# survive Streamlit reruns. Without this, the module-level assignment runs on
# every rerun, creating a fresh dict and destroying any 'running'/'done' state
# set by the background worker — the polling loop never fires, response is lost.
# status: "idle" | "running" | "done" | "stopped" | "error"

@st.cache_resource(show_spinner=False)
def _get_run_state():
    return {
        "status":    "idle",
        "prompt":    "",
        "result":    None,
        "completed": None,
        "error":     None,
        "stop_event": None,
    }, threading.Lock()

_run_state, _run_state_lock = _get_run_state()


def _run_agent_worker(agent, prompt: str, tracer, stop_event: threading.Event) -> None:
    """Background thread: runs the agent and stores result in _run_state."""
    try:
        if tracer:
            result, completed = tracer.run(agent, prompt, stop_event=stop_event)
        else:
            result    = agent.run(prompt, stop_event=stop_event)
            completed = None

        status = "stopped" if stop_event.is_set() else "done"
        with _run_state_lock:
            _run_state.update({
                "status":    status,
                "result":    result,
                "completed": completed,
                "error":     None,
            })
    except Exception as exc:
        import traceback
        print(f"\n[agent worker ERROR]\n{traceback.format_exc()}", flush=True)
        with _run_state_lock:
            _run_state.update({
                "status":    "error",
                "result":    None,
                "completed": None,
                "error":     exc,
            })


from config import Config
from tools import ToolRegistry, register_builtin_tools
from tools.rag import VectorStore, SemanticChunker, register_rag_tools
from tools.arxiv import register_arxiv_tools, search_arxiv_api, fetch_arxiv_paper_api
from tools.dynamic import DynamicToolStore, load_saved_tools, register_dynamic_tools
from tools.router import ToolRouter
from memory import ShortTermMemory
from agent import ReActAgent
from observability.tracer import AgentTracer

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 0; }
    #MainMenu, footer { visibility: hidden; }
    .streamlit-expanderHeader { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── History navigation (↑ / ↓ in the chat input) ─────────────────────────────
# <script> tags inside st.markdown are NOT executed by the browser (HTML5
# innerHTML restriction).  Use components.html() so the script actually runs.
# The iframe is same-origin, so window.parent gives us the main page's DOM.
components.html("""
<script>
(function() {
    // Guard: only set up once per page load (survives Streamlit reruns)
    if (window.parent.__chatHistoryReady) return;
    window.parent.__chatHistoryReady = true;

    const pw  = window.parent;
    const doc = pw.document;

    function setNativeValue(el, value) {
        // Use the native HTMLTextAreaElement setter so React picks up the change
        const nativeSetter = Object.getOwnPropertyDescriptor(
            pw.HTMLTextAreaElement.prototype, 'value'
        ).set;
        nativeSetter.call(el, value);
        el.dispatchEvent(new pw.InputEvent('input',  { bubbles: true, cancelable: true }));
        el.dispatchEvent(new pw.Event('change', { bubbles: true }));
        el.selectionStart = el.selectionEnd = value.length;
    }

    function getHistory() {
        try { return JSON.parse(pw.localStorage.getItem('chatQueryHistory') || '[]'); }
        catch { return []; }
    }

    function saveToHistory(query) {
        if (!query.trim()) return;
        const h = getHistory().filter(q => q !== query);
        h.unshift(query);
        pw.localStorage.setItem('chatQueryHistory', JSON.stringify(h.slice(0, 200)));
    }

    function findTextarea() {
        return (
            doc.querySelector('[data-testid="stChatInputTextArea"]') ||
            doc.querySelector('[data-testid="stChatInput"] textarea') ||
            doc.querySelector('.stChatInput textarea')
        );
    }

    function attachHistory(textarea) {
        if (textarea.dataset.historyEnabled) return;
        textarea.dataset.historyEnabled = 'true';
        let idx = -1;

        textarea.addEventListener('keydown', function(e) {
            const history = getHistory();
            if (e.key === 'ArrowUp') {
                if (this.value === '' || idx >= 0) {
                    e.preventDefault();
                    if (idx < history.length - 1) { idx++; setNativeValue(this, history[idx]); }
                }
            } else if (e.key === 'ArrowDown') {
                if (idx >= 0) {
                    e.preventDefault();
                    idx--;
                    setNativeValue(this, idx >= 0 ? history[idx] : '');
                }
            } else if (e.key === 'Enter' && !e.shiftKey) {
                const val = this.value.trim();
                if (val) saveToHistory(val);
                idx = -1;
            } else {
                if (idx >= 0) idx = -1;
            }
        });
    }

    const observer = new MutationObserver(() => {
        const ta = findTextarea();
        if (ta) attachHistory(ta);
    });
    observer.observe(doc.body, { childList: true, subtree: true });

    const ta = findTextarea();
    if (ta) attachHistory(ta);
})();
</script>
""", height=0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")

# Read toggle states from session state before the sidebar renders
# (widgets set these on the previous run; defaults on first run)
enable_arxiv = st.session_state.get("_enable_arxiv", False)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AI Chatbot")
    st.divider()

    # ── Model
    st.markdown("### Model")
    provider = st.selectbox("Provider", ["anthropic", "openai", "groq", "ollama"])

    model_map = {
        "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
        "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "groq":      ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
        "ollama":    ["llama3.2", "mistral", "deepseek-r1"],
    }
    model = st.selectbox("Model", model_map[provider])

    default_key = {
        "anthropic": get_secret("ANTHROPIC_API_KEY"),
        "openai":    get_secret("OPENAI_API_KEY"),
        "groq":      get_secret("GROQ_API_KEY"),
        "ollama":    "",
    }.get(provider, "")

    if not default_key and provider != "ollama":
        api_key = st.text_input("API Key", type="password",
                                placeholder=f"Enter your {provider} API key")
    else:
        api_key = default_key
        if default_key:
            st.success("API key loaded ✓", icon="🔑")

    st.divider()

    # ── Persona
    st.markdown("### Persona")
    persona = st.selectbox("Chatbot role", [
        "General Assistant", "Customer Support",
        "Data Analyst", "Code Helper", "Custom",
    ])

    _tools_hint = (
        " You have tools available: run_shell (execute terminal commands), "
        "read_file, write_file, search_files, http_request, python_repl. "
        "Use them whenever the user asks about files, directories, running code, or fetching data — "
        "do NOT say you lack access to the local environment. "
        "You also have create_tool — use it when the user wants to save a reusable capability "
        "('add a tool for X', 'save this for later', 'I'll need this again') "
        "or when no existing tool can do the job. "
        "For one-off tasks just use python_repl directly."
    )
    _arxiv_hint = (
        " You also have arXiv tools: search_arxiv (find papers by topic), "
        "fetch_arxiv_papers_batch (download multiple papers IN PARALLEL — use this when you have 2+ IDs), "
        "fetch_arxiv_paper (download a single paper by ID), "
        "search_knowledge_base (search indexed papers), "
        "list_arxiv_papers (show what's indexed). "
        "When the user asks about a research topic or paper: "
        "1) search_arxiv to find relevant papers, "
        "2) fetch_arxiv_papers_batch with ALL paper IDs at once (never loop fetch_arxiv_paper), "
        "3) search_knowledge_base to answer questions from the content. "
        "Always cite the paper ID and title in your answers."
    ) if enable_arxiv else ""
    persona_prompts = {
        "General Assistant": "You are a helpful, friendly AI assistant. Be concise and clear." + _tools_hint + _arxiv_hint,
        "Customer Support":  "You are a professional customer support agent. Be empathetic and solution-focused.",
        "Data Analyst":      "You are an expert data analyst. Use precise language and suggest follow-up analyses." + _tools_hint + _arxiv_hint,
        "Code Helper":       "You are an expert software engineer. Always explain your reasoning." + _tools_hint + _arxiv_hint,
        "Custom":            "",
    }

    if persona == "Custom":
        system_prompt = st.text_area("Custom system prompt",
                                     placeholder="You are a helpful assistant that...", height=100)
    else:
        system_prompt = persona_prompts[persona]
        st.caption(f"*{system_prompt[:80]}{'...' if len(system_prompt) > 80 else ''}*")

    st.divider()

    # ── Tools
    st.markdown("### Tools")
    enable_tools = st.toggle("Enable tools", value=True,
                             help="Shell, HTTP, Python REPL, file ops")
    if enable_tools:
        st.caption("Agent can run shell commands, read/write files, execute Python, fetch URLs")
        _bad_tool_models = {"llama-3.1-8b-instant", "gemma2-9b-it"}
        if model in _bad_tool_models:
            st.warning(
                f"⚠️ `{model}` has unreliable tool-calling and may produce malformed function calls. "
                "Switch to **llama-3.3-70b-versatile**, **claude-haiku-4-5-20251001**, or **gpt-4o-mini** "
                "for stable tool use.",
                icon="⚠️",
            )

    max_iterations = st.slider(
        "Max iterations",
        min_value=3, max_value=25, value=15, step=1,
        help="Maximum ReAct loop iterations per message. "
             "Increase for complex multi-step tasks (e.g. research + RAG). "
             "Decrease to limit cost on simple queries.",
    )

    st.divider()

    # ── arXiv RAG
    st.markdown("### 📄 arXiv Q&A")
    enable_arxiv = st.toggle("Enable arXiv RAG", value=False,
                             key="_enable_arxiv",
                             help="Search and Q&A over arXiv papers")

    if enable_arxiv:
        if provider == "groq" and model == "llama-3.1-8b-instant":
            st.warning(
                "⚠️ llama-3.1-8b has a 6K token/min limit — too small for RAG. "
                "Switch to **llama-3.3-70b-versatile**, **claude-haiku**, or **gpt-4o-mini**.",
                icon="⚠️",
            )
        # Search arXiv
        arxiv_query = st.text_input("Search arXiv papers", placeholder="e.g. retrieval augmented generation")
        if arxiv_query and st.button("Search", key="arxiv_search"):
            with st.spinner("Searching arXiv..."):
                try:
                    papers = search_arxiv_api(arxiv_query, max_results=5)
                    st.session_state["arxiv_search_results"] = papers
                except Exception as e:
                    st.error(f"Search failed: {e}")

        # Show search results with Add buttons
        for p in st.session_state.get("arxiv_search_results", []):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.caption(f"**{p['title'][:55]}{'...' if len(p['title'])>55 else ''}**  \n`{p['id']}` · {p['published'][:7]}")
            with col_b:
                job = _ingest_jobs.get(p["id"], {})
                job_status = job.get("status", "")
                if job_status == "queued":
                    st.caption("⏳")
                elif job_status == "processing":
                    st.caption("⚙️")
                elif job_status == "done":
                    st.caption("✅")
                elif job_status == "error":
                    st.caption("❌")
                    st.caption(job.get("error", "")[:40])
                else:
                    if st.button("Add", key=f"add_{p['id']}"):
                        queue_ingest(p["id"], p["title"], st.session_state["arxiv_store"])
                        st.rerun()

        # Manual add by ID
        arxiv_id = st.text_input("Add by ID", placeholder="e.g. 2005.11401")
        if arxiv_id and st.button("Fetch & Add", key="arxiv_fetch"):
            queue_ingest(arxiv_id.strip(), arxiv_id.strip(), st.session_state["arxiv_store"])
            st.rerun()

        # Ingest progress panel
        with _ingest_lock:
            active_jobs = dict(_ingest_jobs)
        if active_jobs:
            processing = [j for j in active_jobs.values() if j["status"] in ("queued", "processing")]
            done       = [j for j in active_jobs.values() if j["status"] == "done"]
            errors     = [j for j in active_jobs.values() if j["status"] == "error"]
            if processing:
                st.caption(f"⚙️ Ingesting {len(processing)} paper(s)...")
                # Don't interrupt if agent has a result pending — the sidebar rerun
                # fires before the state handler and prevents history from rendering.
                with _run_state_lock:
                    _agent_idle = _run_state["status"] == "idle"
                if _agent_idle:
                    st.rerun()   # auto-refresh while work is in flight
            if done:
                st.caption(f"✅ {len(done)} paper(s) ready")
            if errors:
                for pid, j in active_jobs.items():
                    if j["status"] == "error":
                        st.caption(f"❌ {pid}: {j['error'][:60]}")

        # List indexed papers
        store = st.session_state.get("arxiv_store")
        if store:
            docs = [d for d in store.list_documents() if d["doc_name"].startswith("arxiv:")]
            if docs:
                st.caption(f"**{len(docs)} paper(s) indexed:**")
                for d in docs:
                    st.caption(f"• {d['doc_name'][7:50]}  ({d['chunks']} chunks)")

    st.divider()

    # ── Dynamic Tools
    st.markdown("### 🛠️ Dynamic Tools")
    dyn_store = st.session_state.get("dynamic_tool_store")
    if dyn_store:
        saved = dyn_store.all()
        if saved:
            st.caption(f"{len(saved)} custom tool(s) saved:")
            for tool_name, td in saved.items():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.caption(f"**`{tool_name}`**  \n{td['description'][:60]}")
                with col_b:
                    if st.button("🗑", key=f"del_tool_{tool_name}",
                                 help=f"Delete {tool_name}"):
                        dyn_store.delete(tool_name)
                        st.session_state["agent"] = None
                        st.rerun()
        else:
            st.caption("No custom tools yet. Ask the agent to create one!")

    st.divider()

    # ── Observability (Langfuse)
    st.markdown("### Observability")
    enable_tracing = st.toggle("Enable Langfuse tracing", value=True)

    langfuse_public = get_secret("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = get_secret("LANGFUSE_SECRET_KEY")

    if enable_tracing and not (langfuse_public and langfuse_secret):
        langfuse_public = st.text_input("Langfuse Public Key", type="password",
                                        placeholder="pk-lf-...")
        langfuse_secret = st.text_input("Langfuse Secret Key", type="password",
                                        placeholder="sk-lf-...")
        st.caption("[Get free keys → langfuse.com](https://langfuse.com)")
    elif enable_tracing:
        st.success("Langfuse connected ✓", icon="📊")

    st.divider()

    # ── Session stats
    st.markdown("### Session Stats")
    stats = st.session_state.get("session_stats", {
        "turns": 0, "llm_calls": 0, "tool_calls": 0,
        "total_tokens": 0, "cost_usd": 0.0,
    })

    c1, c2 = st.columns(2)
    c1.metric("Turns",      stats["turns"])
    c2.metric("LLM Calls",  stats["llm_calls"])
    c1.metric("Tool Calls", stats["tool_calls"])
    c2.metric("Tokens",     stats["total_tokens"])
    st.metric("Est. Cost",  f"${stats['cost_usd']:.4f}")

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.agent         = None
        st.session_state.agent_key     = None
        st.session_state.session_stats = {
            "turns": 0, "llm_calls": 0, "tool_calls": 0,
            "total_tokens": 0, "cost_usd": 0.0,
        }
        st.rerun()

    st.caption("Built with custom agentic framework")

# ── Agent builder ─────────────────────────────────────────────────────────────

def build_tracer(session_id: str) -> AgentTracer | None:
    if not enable_tracing or not (langfuse_public and langfuse_secret):
        return None
    try:
        return AgentTracer(
            session_id=session_id,
            model=model,
            provider=provider,
            public_key=langfuse_public,
            secret_key=langfuse_secret,
        )
    except Exception as e:
        st.warning(f"Langfuse init failed: {e}")
        return None


def get_agent(tracer: AgentTracer | None) -> ReActAgent:
    agent_key = f"{provider}|{model}|{system_prompt}|{enable_tools}|{enable_arxiv}|{max_iterations}"

    if st.session_state.get("agent_key") != agent_key or st.session_state.get("agent") is None:
        cfg = Config()
        llm = cfg.build_llm(provider=provider, model=model, api_key=api_key or None)

        tools = ToolRegistry()
        if enable_tools:
            register_builtin_tools(tools)
        if enable_arxiv:
            store = st.session_state["arxiv_store"]
            register_arxiv_tools(tools, store)
            register_rag_tools(tools, store)

        # Always register + reload dynamic tools (persisted across sessions)
        dyn_store = st.session_state["dynamic_tool_store"]
        load_saved_tools(tools, dyn_store)
        register_dynamic_tools(tools, dyn_store)

        # Use tracer callbacks if available — gives per-LLM + per-tool spans
        callbacks = tracer.make_callbacks() if tracer else {
            "on_thought":     lambda i, t: None,
            "on_tool_call":   lambda i, tc: None,
            "on_tool_result": lambda i, r: None,
            "on_final":       lambda a: None,
        }

        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=ShortTermMemory(max_messages=50),
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            tool_router=ToolRouter(max_tools=10),
            **callbacks,
        )
        st.session_state.agent     = agent
        st.session_state.agent_key = agent_key

    return st.session_state.agent

# ── Session init ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "turns": 0, "llm_calls": 0, "tool_calls": 0,
        "total_tokens": 0, "cost_usd": 0.0,
    }
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "dynamic_tool_store" not in st.session_state:
    st.session_state["dynamic_tool_store"] = DynamicToolStore(
        tools_file=".agent_tools/tools.json"
    )
if "arxiv_store" not in st.session_state:
    st.session_state["arxiv_store"] = VectorStore(
        persist_dir=".arxiv_store",
        chunker=SemanticChunker(target_size=600, overlap_sentences=2),
    )
if "arxiv_search_results" not in st.session_state:
    st.session_state["arxiv_search_results"] = []

# ── Chat header ───────────────────────────────────────────────────────────────

h_col, b_col = st.columns([5, 1])
with h_col:
    st.markdown(f"### {persona}")
with b_col:
    colors = {"anthropic": "🟠", "openai": "🟢", "groq": "🔵", "ollama": "⚫"}
    st.markdown(f"{colors.get(provider,'⚪')} `{model.split('-')[0]}`")

# ── Background run state handler ──────────────────────────────────────────────

with _run_state_lock:
    _rs = dict(_run_state)   # snapshot

print(f"[Streamlit] script rerun — _run_state status={_rs['status']!r}, msgs={len(st.session_state.get('messages', []))}", flush=True)

if _rs["status"] == "running":
    # Render existing history (user message is already in session_state.messages)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.chat_message("assistant"):
        col_think, col_stop = st.columns([5, 1])
        with col_think:
            st.markdown("⏳ _Thinking…_")
        with col_stop:
            if st.button("⏹ Stop", key="stop_run", help="Stop the agent mid-run"):
                with _run_state_lock:
                    ev = _run_state.get("stop_event")
                    if ev:
                        ev.set()

    time.sleep(0.5)   # poll interval
    st.rerun()

elif _rs["status"] in ("done", "stopped", "error"):
    # ── Reset module state immediately so future reruns see "idle" ──────────
    with _run_state_lock:
        _run_state.update({
            "status": "idle", "prompt": "", "result": None,
            "completed": None, "error": None, "stop_event": None,
        })

    result    = _rs["result"]
    completed = _rs["completed"]
    error     = _rs["error"]
    status    = _rs["status"]

    print(f"[Streamlit] Run finished: status={status}, has_result={result is not None}, has_error={error is not None}", flush=True)

    try:
        if status == "error":
            err_str = str(error) if error else ""
            if "tool_use_failed" in err_str or ("400" in err_str and "failed_generation" in err_str):
                _alts = [m for m in ["claude-haiku-4-5-20251001", "gpt-4o-mini", "llama-3.3-70b-versatile"] if m != model]
                err = (
                    f"**`{model}` generated a malformed tool call and was rejected by the API.**\n\n"
                    "This model sometimes outputs tool calls in the wrong format. "
                    "Even large models can hit this on complex tool schemas.\n\n"
                    "**Try one of these instead:**\n"
                    + "".join(f"- `{m}`\n" for m in _alts) +
                    "\nOr **disable Tools** in the sidebar if you don't need them for this query."
                )
            elif "413" in err_str or "too large" in err_str.lower():
                err = (
                    "**Request too large for this model.**\n\n"
                    "- Switch to a model with higher limits\n"
                    "- Click **Clear chat** in the sidebar\n"
                    "- Disable arXiv RAG if active"
                )
            elif "401" in err_str or "authentication" in err_str.lower() or "api_key" in err_str.lower():
                err = f"**Invalid API key.** Please check your {provider} API key in the sidebar."
            elif "429" in err_str or "rate limit" in err_str.lower() or "rate_limit_exceeded" in err_str:
                import re as _re
                _wait = _re.search(r"try again in ([\dm\s\.]+s)", err_str)
                _wait_str = f" Try again in **{_wait.group(1).strip()}**." if _wait else ""
                _tpd = "tokens per day" in err_str.lower() or "tpd" in err_str
                _limit_note = (
                    " This is the **daily token limit (TPD)** — separate from per-minute limits."
                    " Upgrade to Groq Dev Tier for higher limits."
                    if _tpd else ""
                )
                err = f"**Rate limit hit for `{model}`.**{_wait_str}{_limit_note}"
            else:
                err = f"Something went wrong: {error}"
            st.session_state.messages.append({"role": "assistant", "content": err})

        elif result:
            trace_info = None
            if completed:
                trace_info = {
                    "answered_directly": completed.answered_directly,
                    "llm_calls":         completed.llm_calls,
                    "tool_calls":        completed.tool_calls,
                    "tokens":            completed.prompt_tokens + completed.completion_tokens,
                    "cost":              completed.cost_usd,
                    "latency_ms":        completed.latency_ms,
                    "trace_id":          completed.trace_id,
                }
                s = st.session_state.session_stats
                s["turns"]        += 1
                s["llm_calls"]    += completed.llm_calls
                s["tool_calls"]   += completed.tool_calls
                s["total_tokens"] += trace_info["tokens"]
                s["cost_usd"]     += completed.cost_usd
            else:
                st.session_state.session_stats["turns"] += 1

            tool_steps = []
            for step in result.steps:
                for tc, tr in zip(step.tool_calls, step.tool_results):
                    tool_steps.append({
                        "tool": tc.name, "args": tc.arguments,
                        "result": tr.content, "is_error": tr.is_error,
                    })

            content = result.answer
            if status == "stopped":
                content = (
                    f"⏹ _Stopped early._\n\n{result.answer}"
                    if result.answer and result.answer != "Processing was stopped."
                    else "⏹ _Processing was stopped._"
                )

            st.session_state.messages.append({
                "role":       "assistant",
                "content":    content,
                "trace":      trace_info,
                "tool_steps": tool_steps,
            })
            print(f"[Streamlit] Message appended. Total messages: {len(st.session_state.messages)}", flush=True)

        else:
            print(f"[Streamlit] WARNING: status={status} but result=None and error=None", flush=True)

    except Exception as _done_exc:
        import traceback as _tb
        print(f"[Streamlit done-handler ERROR]\n{_tb.format_exc()}", flush=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"An internal error occurred while displaying the response: {_done_exc}",
        })
    # Trigger a clean "idle" rerun so the browser renders the final message.
    # Without this, Streamlit does not reliably flush the delta to the browser
    # when transitioning from the "running" polling loop to the "done" state.
    st.rerun()

# ── Render history ────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Trace info badge
        if msg.get("trace"):
            t = msg["trace"]
            mode_icon = "💬" if t["answered_directly"] else "🔧"
            mode_label = "Direct answer" if t["answered_directly"] else f"Used framework ({t['tool_calls']} tools)"
            st.caption(
                f"{mode_icon} {mode_label} · "
                f"LLM calls: {t['llm_calls']} · "
                f"Tokens: {t['tokens']} · "
                f"Cost: ${t['cost']:.4f} · "
                f"Latency: {t['latency_ms']}ms"
            )

        if msg.get("tool_steps") and enable_tools:
            with st.expander(f"🔧 Tool calls ({len(msg['tool_steps'])})", expanded=False):
                for step in msg["tool_steps"]:
                    st.markdown(f"**`{step['tool']}`**")
                    st.code(str(step["args"]), language="python")
                    icon = "❌" if step["is_error"] else "✅"
                    st.caption(f"{icon} {step['result'][:300]}")

# ── Empty state starters ──────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    starters = {
        "General Assistant": ["What can you help me with?", "Tell me something interesting", "Explain quantum computing simply"],
        "Customer Support":  ["I have an issue with my order", "How do I get a refund?", "I can't log into my account"],
        "Data Analyst":      ["How do I calculate churn rate?", "Explain cohort analysis", "What metrics should I track?"],
        "Code Helper":       ["Review this Python function", "What's the best way to handle errors?", "Explain async/await"],
        "Custom":            ["Hello!", "What can you do?", "Help me with a task"],
    }
    cols = st.columns(3)
    for i, s in enumerate(starters.get(persona, starters["General Assistant"])):
        with cols[i]:
            if st.button(s, use_container_width=True):
                st.session_state._starter = s
                st.rerun()

prompt = None
if hasattr(st.session_state, "_starter"):
    prompt = st.session_state._starter
    del st.session_state._starter

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Type your message here...")
if user_input:
    prompt = user_input

if prompt:
    if not api_key and provider != "ollama":
        st.error(f"Please enter your {provider} API key in the sidebar.")
        st.stop()

    # Don't accept a new message while one is already in flight
    with _run_state_lock:
        already_running = _run_state["status"] == "running"
    if already_running:
        st.warning("Please wait — the agent is still processing the previous message.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        tracer     = build_tracer(st.session_state.session_id)
        agent      = get_agent(tracer)
        stop_event = threading.Event()

        with _run_state_lock:
            _run_state.update({
                "status":     "running",
                "prompt":     prompt,
                "result":     None,
                "completed":  None,
                "error":      None,
                "stop_event": stop_event,
            })

        threading.Thread(
            target=_run_agent_worker,
            args=(agent, prompt, tracer, stop_event),
            daemon=True,
        ).start()

    except Exception as _setup_exc:
        import traceback as _tb
        print(f"[AGENT SETUP ERROR]\n{_tb.format_exc()}", flush=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Failed to start agent:** {_setup_exc}",
        })

    st.rerun()
