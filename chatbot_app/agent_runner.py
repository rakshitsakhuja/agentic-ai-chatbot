"""Agent construction and background worker helpers."""

import threading
import streamlit as st

from config import Config
from tools import ToolRegistry, register_builtin_tools
from tools.rag import register_rag_tools
from tools.arxiv import register_arxiv_tools
from tools.dynamic import load_saved_tools, register_dynamic_tools
from tools.router import ToolRouter
from memory import ShortTermMemory
from agent import ReActAgent
from observability.tracer import AgentTracer


def _bg_ingest_worker(paper_id: str, store, ingest_jobs: dict, ingest_lock) -> None:
    from tools.arxiv import fetch_arxiv_paper_api, _fetch_html_text, _fetch_pdf_text, build_paper_text
    with ingest_lock:
        ingest_jobs[paper_id]["status"] = "processing"
    try:
        paper = fetch_arxiv_paper_api(paper_id)
        if not paper:
            with ingest_lock:
                ingest_jobs[paper_id].update({"status": "error", "error": "Paper not found"})
            return
        full_text = _fetch_html_text(paper["id"]) or _fetch_pdf_text(paper["pdf_url"])
        text = build_paper_text(paper, full_text)
        doc_name = f"arxiv:{paper['id']} - {paper['title'][:60]}"
        store.ingest_text(text, doc_name, {"arxiv_id": paper["id"], "title": paper["title"]})
        with ingest_lock:
            ingest_jobs[paper_id].update({"status": "done", "title": paper["title"][:50]})
    except Exception as e:
        import traceback
        print(f"\n[ingest worker ERROR] {paper_id}\n{traceback.format_exc()}", flush=True)
        with ingest_lock:
            ingest_jobs[paper_id].update({"status": "error", "error": str(e)})


def queue_ingest(paper_id: str, title: str, store, ingest_jobs: dict, ingest_lock) -> None:
    with ingest_lock:
        if paper_id in ingest_jobs and ingest_jobs[paper_id]["status"] in ("queued", "processing"):
            return
        ingest_jobs[paper_id] = {"status": "queued", "title": title, "error": ""}
    t = threading.Thread(
        target=_bg_ingest_worker,
        args=(paper_id, store, ingest_jobs, ingest_lock),
        daemon=True,
    )
    t.start()


def run_agent_worker(agent, prompt: str, tracer, stop_event: threading.Event, run_state: dict, run_state_lock) -> None:
    try:
        if tracer:
            result, completed = tracer.run(agent, prompt, stop_event=stop_event)
        else:
            result    = agent.run(prompt, stop_event=stop_event)
            completed = None

        status = "stopped" if stop_event.is_set() else "done"
        with run_state_lock:
            run_state.update({
                "status": status,
                "result": result,
                "completed": completed,
                "error": None,
            })
    except Exception as exc:
        import traceback
        print(f"\n[agent worker ERROR]\n{traceback.format_exc()}", flush=True)
        with run_state_lock:
            run_state.update({
                "status": "error",
                "result": None,
                "completed": None,
                "error": exc,
            })


def build_tracer(session_id: str, settings: dict) -> AgentTracer | None:
    if not settings["enable_tracing"] or not (settings["langfuse_public"] and settings["langfuse_secret"]):
        return None
    try:
        return AgentTracer(
            session_id=session_id,
            model=settings["model"],
            provider=settings["provider"],
            public_key=settings["langfuse_public"],
            secret_key=settings["langfuse_secret"],
        )
    except Exception as e:
        st.warning(f"Langfuse init failed: {e}")
        return None


def get_agent(tracer: AgentTracer | None, settings: dict) -> ReActAgent:
    agent_key = (
        f"{settings['provider']}|{settings['model']}|{settings['system_prompt']}|"
        f"{settings['enable_tools']}|{settings['enable_arxiv']}|{settings['max_iterations']}"
    )

    if st.session_state.get("agent_key") != agent_key or st.session_state.get("agent") is None:
        cfg = Config()
        llm = cfg.build_llm(
            provider=settings["provider"],
            model=settings["model"],
            api_key=settings["api_key"] or None,
        )

        tools = ToolRegistry()
        if settings["enable_tools"]:
            register_builtin_tools(tools)
        if settings["enable_arxiv"]:
            store = st.session_state["arxiv_store"]
            register_arxiv_tools(tools, store)
            register_rag_tools(tools, store)

        dyn_store = st.session_state["dynamic_tool_store"]
        load_saved_tools(tools, dyn_store)
        register_dynamic_tools(tools, dyn_store)

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
            system_prompt=settings["system_prompt"],
            max_iterations=settings["max_iterations"],
            tool_router=ToolRouter(max_tools=10),
            **callbacks,
        )
        st.session_state.agent     = agent
        st.session_state.agent_key = agent_key

    return st.session_state.agent
