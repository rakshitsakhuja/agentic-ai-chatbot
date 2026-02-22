"""Shared configuration defaults and model lists."""

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

MODEL_MAP = {
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
    "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "groq":      ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
    "ollama":    ["llama3.2", "mistral", "deepseek-r1"],
}

BAD_TOOL_MODELS = {"llama-3.1-8b-instant", "gemma2-9b-it"}
TOOL_CALL_ALTERNATIVES = [
    "claude-haiku-4-5-20251001",
    "gpt-4o-mini",
    "llama-3.3-70b-versatile",
]

DEFAULT_SESSION_STATS = {
    "turns": 0,
    "llm_calls": 0,
    "tool_calls": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}
