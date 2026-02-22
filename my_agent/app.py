"""
Streamlit Chatbot UI
─────────────────────
Deploy free on Streamlit Cloud via GitHub.

Run locally:
    pip install streamlit
    streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from tools import ToolRegistry, register_builtin_tools
from memory import ShortTermMemory
from agent import ReActAgent

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="My AI Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 My AI Agent")
st.caption("Powered by your custom agentic framework")

# ── Sidebar — settings ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    provider = st.selectbox(
        "LLM Provider",
        ["anthropic", "openai", "groq", "ollama"],
        index=0,
    )

    model_defaults = {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai":    "gpt-4o-mini",
        "groq":      "llama-3.1-8b-instant",
        "ollama":    "llama3.2",
    }
    model = st.text_input("Model", value=model_defaults[provider])

    api_key = st.text_input(
        "API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Or set via environment variable",
    )

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant. Be concise and friendly.",
        height=120,
    )

    enable_tools = st.toggle("Enable built-in tools", value=False,
                             help="Shell, file ops, HTTP, Python REPL")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()

    st.divider()
    st.caption("Built with custom agentic framework")

# ── Agent initialisation (cached per session) ─────────────────────────────────

def build_agent(provider, model, api_key, system_prompt, enable_tools):
    cfg = Config()

    llm = cfg.build_llm(
        provider=provider,
        model=model,
        api_key=api_key if api_key else None,
    )

    tools = ToolRegistry()
    if enable_tools:
        register_builtin_tools(tools)

    memory = ShortTermMemory(max_messages=40)

    return ReActAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        system_prompt=system_prompt,
        max_iterations=8,
        # Silent callbacks — Streamlit handles display
        on_thought=lambda i, t: None,
        on_tool_call=lambda i, tc: None,
        on_tool_result=lambda i, r: None,
        on_final=lambda a: None,
    )


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

# Rebuild agent if settings change
agent_key = f"{provider}:{model}:{system_prompt}:{enable_tools}"
if st.session_state.get("agent_key") != agent_key:
    st.session_state.agent_key = agent_key
    st.session_state.agent = None   # force rebuild on next message

# ── Chat display ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask me anything..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build agent lazily (only when first message arrives)
    if st.session_state.agent is None:
        if not api_key and provider != "ollama":
            st.error("Please enter your API key in the sidebar.")
            st.stop()
        try:
            st.session_state.agent = build_agent(
                provider, model, api_key, system_prompt, enable_tools
            )
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")
            st.stop()

    agent: ReActAgent = st.session_state.agent

    # Run agent and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.run(prompt)
                answer = result.answer

                st.markdown(answer)

                # Show tool usage in expander if tools were used
                if result.steps and enable_tools:
                    with st.expander(f"🔧 {len(result.steps)} tool call(s)", expanded=False):
                        for step in result.steps:
                            for tc in step.tool_calls:
                                st.code(f"→ {tc.name}({tc.arguments})", language="python")
                            for tr in step.tool_results:
                                status = "❌" if tr.is_error else "✅"
                                st.text(f"{status} {tr.content[:300]}")

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
