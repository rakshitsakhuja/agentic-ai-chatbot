"""
Configuration
──────────────
Centralised config — loaded from environment variables.
Override any value by setting the env var or passing kwargs.

Usage:
    from config import Config
    cfg = Config()
    llm = cfg.build_llm()
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── LLM provider ──────────────────────────────────────────────────────────
    # "openai" | "anthropic" | "groq" | "ollama" | "deepseek" | "together"
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))

    # Model name — provider-specific
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"))

    # API keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Custom base URL (for Ollama, LM Studio, proxies, etc.)
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL", None))

    # ── Agent behaviour ───────────────────────────────────────────────────────
    max_iterations: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_ITER", "15")))
    temperature: float = field(default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.0")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "4096")))

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_dir: str = field(default_factory=lambda: os.getenv("MEMORY_DIR", ".agent_memory"))
    short_term_window: int = field(default_factory=lambda: int(os.getenv("MEMORY_WINDOW", "40")))

    # ── Reflection ────────────────────────────────────────────────────────────
    enable_reflection: bool = field(default_factory=lambda: os.getenv("ENABLE_REFLECTION", "false").lower() == "true")
    reflection_max_retries: int = field(default_factory=lambda: int(os.getenv("REFLECTION_RETRIES", "2")))

    def build_llm(self, **overrides):
        """Instantiate the configured LLM provider."""
        from llm.openai_llm import OpenAILLM
        from llm.anthropic_llm import AnthropicLLM

        provider = overrides.get("provider", self.provider).lower()
        model = overrides.get("model", self.model)

        BASE_URLS = {
            "groq":     "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "ollama":   "http://localhost:11434/v1",
        }

        if provider == "anthropic":
            return AnthropicLLM(
                model=model,
                api_key=overrides.get("api_key", self.anthropic_api_key),
            )

        # All OpenAI-compatible providers
        api_key = overrides.get("api_key") or {
            "openai":   self.openai_api_key,
            "groq":     self.groq_api_key,
            "ollama":   "ollama",
        }.get(provider, "EMPTY")

        base_url = overrides.get("base_url") or self.base_url or BASE_URLS.get(provider)

        return OpenAILLM(model=model, api_key=api_key, base_url=base_url)
