"""UI helpers for rendering chat history and formatting errors."""

import streamlit as st

from constants import TOOL_CALL_ALTERNATIVES


def format_run_error(err: Exception | None, model: str, provider: str) -> str:
    err_str = str(err) if err else ""
    if "tool_use_failed" in err_str or ("400" in err_str and "failed_generation" in err_str):
        _alts = [m for m in TOOL_CALL_ALTERNATIVES if m != model]
        return (
            f"**`{model}` generated a malformed tool call and was rejected by the API.**\n\n"
            "This model sometimes outputs tool calls in the wrong format. "
            "Even large models can hit this on complex tool schemas.\n\n"
            "**Try one of these instead:**\n"
            + "".join(f"- `{m}`\n" for m in _alts) +
            "\nOr **disable Tools** in the sidebar if you don't need them for this query."
        )
    if "413" in err_str or "too large" in err_str.lower():
        return (
            "**Request too large for this model.**\n\n"
            "- Switch to a model with higher limits\n"
            "- Click **Clear chat** in the sidebar\n"
            "- Disable arXiv RAG if active"
        )
    if "401" in err_str or "authentication" in err_str.lower() or "api_key" in err_str.lower():
        return f"**Invalid API key.** Please check your {provider} API key in the sidebar."
    if "429" in err_str or "rate limit" in err_str.lower() or "rate_limit_exceeded" in err_str:
        import re as _re
        _wait = _re.search(r"try again in ([\dm\s\.]+s)", err_str)
        _wait_str = f" Try again in **{_wait.group(1).strip()}**." if _wait else ""
        _tpd = "tokens per day" in err_str.lower() or "tpd" in err_str
        _limit_note = (
            " This is the **daily token limit (TPD)** — separate from per-minute limits."
            " Upgrade to Groq Dev Tier for higher limits."
            if _tpd else ""
        )
        return f"**Rate limit hit for `{model}`.**{_wait_str}{_limit_note}"
    return f"Something went wrong: {err}"


def collect_tool_steps(result) -> list[dict]:
    tool_steps = []
    for step in result.steps:
        for tc, tr in zip(step.tool_calls, step.tool_results):
            tool_steps.append({
                "tool": tc.name,
                "args": tc.arguments,
                "result": tr.content,
                "is_error": tr.is_error,
            })
    return tool_steps


def render_chat_history(messages: list[dict], enable_tools: bool) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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


def render_empty_state(persona: str) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    starters = {
        "General Assistant": [
            "What can you help me with?",
            "Tell me something interesting",
            "Explain quantum computing simply",
        ],
        "Customer Support": [
            "I have an issue with my order",
            "How do I get a refund?",
            "I can't log into my account",
        ],
        "Data Analyst": [
            "How do I calculate churn rate?",
            "Explain cohort analysis",
            "What metrics should I track?",
        ],
        "Code Helper": [
            "Review this Python function",
            "What's the best way to handle errors?",
            "Explain async/await",
        ],
        "Custom": ["Hello!", "What can you do?", "Help me with a task"],
    }
    cols = st.columns(3)
    for i, s in enumerate(starters.get(persona, starters["General Assistant"])):
        with cols[i]:
            if st.button(s, use_container_width=True):
                st.session_state._starter = s
                st.rerun()


def render_header(persona: str, provider: str, model: str) -> None:
    h_col, b_col = st.columns([5, 1])
    with h_col:
        st.markdown(f"### {persona}")
    with b_col:
        colors = {"anthropic": "🟠", "openai": "🟢", "groq": "🔵", "ollama": "⚫"}
        st.markdown(f"{colors.get(provider,'⚪')} `{model.split('-')[0]}`")
