"""
Entry Point
────────────
Wires everything together and runs an interactive agent REPL.

Quick start:
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py

Or with any OpenAI-compatible model:
    export LLM_PROVIDER=groq
    export LLM_MODEL=llama-3.1-8b-instant
    export GROQ_API_KEY=gsk_...
    python main.py

Or Ollama (no API key needed):
    export LLM_PROVIDER=ollama
    export LLM_MODEL=llama3.2
    python main.py
"""

import sys
from config import Config
from tools import ToolRegistry, register_builtin_tools
from memory import ShortTermMemory, KVStore, EpisodicStore
from agent import ReActAgent, Reflector


def build_agent(cfg: Config) -> ReActAgent:
    """Wire up LLM + tools + memory → return ready-to-use agent."""
    llm = cfg.build_llm()

    # Tool registry — add your domain tools here
    tools = ToolRegistry()
    register_builtin_tools(tools)
    # register_trading_tools(tools)  ← plug in your custom tools

    # Memory
    memory = ShortTermMemory(max_messages=cfg.short_term_window)

    # Agent
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        max_iterations=cfg.max_iterations,
    )

    return agent


def run_single(task: str, cfg: Config):
    """Run a single task and exit."""
    agent = build_agent(cfg)

    if cfg.enable_reflection:
        reflector = Reflector(cfg.build_llm(), max_retries=cfg.reflection_max_retries)
        initial = agent.run(task)
        result = reflector.reflect_and_refine(agent, task, initial)
        print(f"\n{'='*60}")
        print(f"Final answer (attempts={result.attempts}, score={result.score}/10):")
        print(result.final_answer)
    else:
        result = agent.run(task)
        print(f"\nTokens used: {result.total_tokens} | Steps: {len(result.steps)}")


def run_repl(cfg: Config):
    """Interactive REPL loop."""
    agent = build_agent(cfg)

    print(f"\n Agent ready | provider={cfg.provider} model={cfg.model}")
    print(" Commands: 'reset' to clear memory, 'exit' to quit, 'tools' to list tools\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Bye.")
            break
        elif user_input.lower() == "reset":
            agent.reset()
            print("[Memory cleared]")
            continue
        elif user_input.lower() == "tools":
            print("Available tools:", agent.tools.list_tools())
            continue

        agent.run(user_input)


def main():
    cfg = Config()

    # Single task from CLI args
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        run_single(task, cfg)
    else:
        run_repl(cfg)


if __name__ == "__main__":
    main()
