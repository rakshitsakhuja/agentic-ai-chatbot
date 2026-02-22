"""Session state and cached run/ingest state helpers."""

import os
import threading
import streamlit as st

from constants import DEFAULT_SESSION_STATS
from tools.dynamic import DynamicToolStore
from tools.rag import VectorStore, SemanticChunker


def get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:
        return os.getenv(key, "")


def default_session_stats() -> dict:
    return dict(DEFAULT_SESSION_STATS)


@st.cache_resource(show_spinner=False)
def get_ingest_state():
    return {}, threading.Lock()


@st.cache_resource(show_spinner=False)
def get_run_state():
    return {
        "status": "idle",
        "prompt": "",
        "result": None,
        "completed": None,
        "error": None,
        "stop_event": None,
    }, threading.Lock()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = default_session_stats()
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
