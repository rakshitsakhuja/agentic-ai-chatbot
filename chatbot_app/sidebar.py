"""Sidebar rendering and settings assembly."""

import streamlit as st

from constants import BAD_TOOL_MODELS, DEFAULT_MODEL, DEFAULT_PROVIDER, MODEL_MAP
from agent_runner import queue_ingest
from state import default_session_stats, get_secret
from tools.arxiv import search_arxiv_api


def render_sidebar(enable_arxiv_default: bool, ingest_jobs: dict, ingest_lock, run_state: dict, run_state_lock) -> dict:
    with st.sidebar:
        st.markdown("## 🤖 AI Chatbot")
        st.divider()

        st.markdown("### Model")
        provider_options = ["anthropic", "openai", "groq", "ollama"]
        provider_index = provider_options.index(DEFAULT_PROVIDER)
        provider = st.selectbox("Provider", provider_options, index=provider_index)

        model_options = MODEL_MAP[provider]
        model_index = 0
        if provider == DEFAULT_PROVIDER and DEFAULT_MODEL in model_options:
            model_index = model_options.index(DEFAULT_MODEL)
        model = st.selectbox("Model", model_options, index=model_index)

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
        ) if enable_arxiv_default else ""
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

        st.markdown("### Tools")
        enable_tools = st.toggle("Enable tools", value=True,
                                 help="Shell, HTTP, Python REPL, file ops")
        if enable_tools:
            st.caption("Agent can run shell commands, read/write files, execute Python, fetch URLs")
            if model in BAD_TOOL_MODELS:
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
            arxiv_query = st.text_input("Search arXiv papers", placeholder="e.g. retrieval augmented generation")
            if arxiv_query and st.button("Search", key="arxiv_search"):
                with st.spinner("Searching arXiv..."):
                    try:
                        papers = search_arxiv_api(arxiv_query, max_results=5)
                        st.session_state["arxiv_search_results"] = papers
                    except Exception as e:
                        st.error(f"Search failed: {e}")

            for p in st.session_state.get("arxiv_search_results", []):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.caption(f"**{p['title'][:55]}{'...' if len(p['title'])>55 else ''}**  \n`{p['id']}` · {p['published'][:7]}")
                with col_b:
                    job = ingest_jobs.get(p["id"], {})
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
                            queue_ingest(p["id"], p["title"], st.session_state["arxiv_store"], ingest_jobs, ingest_lock)
                            st.rerun()

            arxiv_id = st.text_input("Add by ID", placeholder="e.g. 2005.11401")
            if arxiv_id and st.button("Fetch & Add", key="arxiv_fetch"):
                queue_ingest(arxiv_id.strip(), arxiv_id.strip(), st.session_state["arxiv_store"], ingest_jobs, ingest_lock)
                st.rerun()

            with ingest_lock:
                active_jobs = dict(ingest_jobs)
            if active_jobs:
                processing = [j for j in active_jobs.values() if j["status"] in ("queued", "processing")]
                done       = [j for j in active_jobs.values() if j["status"] == "done"]
                errors     = [j for j in active_jobs.values() if j["status"] == "error"]
                if processing:
                    st.caption(f"⚙️ Ingesting {len(processing)} paper(s)...")
                    with run_state_lock:
                        agent_idle = run_state["status"] == "idle"
                    if agent_idle:
                        st.rerun()
                if done:
                    st.caption(f"✅ {len(done)} paper(s) ready")
                if errors:
                    for pid, j in active_jobs.items():
                        if j["status"] == "error":
                            st.caption(f"❌ {pid}: {j['error'][:60]}")

            store = st.session_state.get("arxiv_store")
            if store:
                docs = [d for d in store.list_documents() if d["doc_name"].startswith("arxiv:")]
                if docs:
                    st.caption(f"**{len(docs)} paper(s) indexed:**")
                    for d in docs:
                        st.caption(f"• {d['doc_name'][7:50]}  ({d['chunks']} chunks)")

        st.divider()

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

        st.markdown("### Session Stats")
        stats = st.session_state.get("session_stats", default_session_stats())

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
            st.session_state.session_stats = default_session_stats()
            st.rerun()

        st.caption("Built with custom agentic framework")

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "persona": persona,
        "system_prompt": system_prompt,
        "enable_tools": enable_tools,
        "enable_arxiv": enable_arxiv,
        "max_iterations": max_iterations,
        "enable_tracing": enable_tracing,
        "langfuse_public": langfuse_public,
        "langfuse_secret": langfuse_secret,
    }
