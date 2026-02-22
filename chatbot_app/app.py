"""
Chatbot App — Streamlit UI with Langfuse Tracing
──────────────────────────────────────────────────
Top-level app orchestration and Streamlit event loop.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Cloud → done
"""

import os
import sys

# ── Import agent framework ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "my_agent"))

import time
import threading
import streamlit as st
import streamlit.components.v1 as components

from agent_runner import build_tracer, get_agent, run_agent_worker
from sidebar import render_sidebar
from state import get_ingest_state, get_run_state, init_session_state
from ui import (
    collect_tool_steps,
    format_run_error,
    render_chat_history,
    render_empty_state,
    render_header,
)

_ingest_jobs, _ingest_lock = get_ingest_state()
_run_state, _run_state_lock = get_run_state()

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


# ── Session init ──────────────────────────────────────────────────────────────

init_session_state()

enable_arxiv_default = st.session_state.get("_enable_arxiv", False)
settings = render_sidebar(enable_arxiv_default, _ingest_jobs, _ingest_lock, _run_state, _run_state_lock)
provider = settings["provider"]
model = settings["model"]
api_key = settings["api_key"]
persona = settings["persona"]
system_prompt = settings["system_prompt"]
enable_tools = settings["enable_tools"]
enable_arxiv = settings["enable_arxiv"]
max_iterations = settings["max_iterations"]

# ── Chat header ───────────────────────────────────────────────────────────────

render_header(persona, provider, model)

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
            err = format_run_error(error, model, provider)
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

            tool_steps = collect_tool_steps(result)

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

render_chat_history(st.session_state.messages, enable_tools)

# ── Empty state starters ──────────────────────────────────────────────────────

if not st.session_state.messages:
    render_empty_state(persona)

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
        tracer     = build_tracer(st.session_state.session_id, settings)
        agent      = get_agent(tracer, settings)
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
            target=run_agent_worker,
            args=(agent, prompt, tracer, stop_event, _run_state, _run_state_lock),
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
