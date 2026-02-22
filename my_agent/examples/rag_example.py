"""
RAG Application Example
────────────────────────
Shows 3 ways to build RAG with the boilerplate:

  1. Simple RAG     — agent searches docs to answer questions
  2. Agentic RAG    — agent decides WHEN to search (not always)
  3. Multi-doc RAG  — agent searches across multiple sources + reflects
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from tools import ToolRegistry, register_builtin_tools
from tools.rag import VectorStore, register_rag_tools
from memory import ShortTermMemory
from agent import ReActAgent, Reflector


# ── 1. Simple RAG ─────────────────────────────────────────────────────────────

def example_simple_rag():
    """
    Ingest some docs, then ask the agent questions.
    Agent automatically calls search_knowledge_base before answering.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple RAG")
    print("="*60)

    cfg = Config()
    llm = cfg.build_llm()

    # Setup
    store = VectorStore(persist_dir=".agent_memory/rag_simple")
    tools = ToolRegistry()
    register_builtin_tools(tools)
    register_rag_tools(tools, store)

    # Ingest documents (run once, persisted to disk)
    store.ingest_text(
        text="""
        Bitcoin (BTC) was created in 2009 by Satoshi Nakamoto.
        It has a maximum supply of 21 million coins.
        Bitcoin uses Proof of Work consensus mechanism.
        The block time is approximately 10 minutes.
        Bitcoin halving occurs every 210,000 blocks, roughly every 4 years.
        The last halving was in April 2024, reducing block reward to 3.125 BTC.
        """,
        doc_name="bitcoin_basics.txt",
    )

    store.ingest_text(
        text="""
        Ethereum (ETH) was created by Vitalik Buterin and launched in 2015.
        Ethereum supports smart contracts and decentralised applications (dApps).
        In 2022, Ethereum merged from Proof of Work to Proof of Stake (The Merge).
        Ethereum uses EIP-1559 fee mechanism with base fee burning.
        The Ethereum Virtual Machine (EVM) is the runtime for smart contracts.
        Layer 2 solutions like Arbitrum and Optimism scale Ethereum.
        """,
        doc_name="ethereum_basics.txt",
    )

    RAG_PROMPT = """You are a knowledgeable crypto assistant.
ALWAYS call search_knowledge_base FIRST before answering any question.
Base your answers strictly on what you find in the knowledge base.
If the answer is not in the knowledge base, say so clearly."""

    agent = ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=RAG_PROMPT,
    )

    questions = [
        "What is Bitcoin's maximum supply and when was it created?",
        "What consensus mechanism does Ethereum use after The Merge?",
        "How often does Bitcoin halving occur?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = agent.run(q)
        agent.reset()  # fresh context per question


# ── 2. Agentic RAG ────────────────────────────────────────────────────────────

def example_agentic_rag():
    """
    Agent decides intelligently when to search vs when to answer directly.
    Multi-hop: agent searches, finds partial answer, searches again with new query.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Agentic RAG (multi-hop)")
    print("="*60)

    cfg = Config()
    llm = cfg.build_llm()

    store = VectorStore(persist_dir=".agent_memory/rag_agentic")
    tools = ToolRegistry()
    register_builtin_tools(tools)
    register_rag_tools(tools, store)

    store.ingest_text(
        text="""
        Q4 2024 Revenue: $12.5M (up 40% YoY)
        Q3 2024 Revenue: $10.2M
        Q2 2024 Revenue: $8.9M
        Q1 2024 Revenue: $7.1M
        Full year 2024 total revenue: $38.7M
        Top product: Pro Plan — $8.2M revenue in 2024
        Enterprise customers grew from 45 to 89 in 2024
        """,
        doc_name="revenue_report_2024.txt",
    )

    store.ingest_text(
        text="""
        2025 Growth targets:
        - Revenue target: $60M (55% growth)
        - Enterprise customers target: 200
        - New markets: Europe and Southeast Asia
        - Product launches: AI features in Q2, Mobile app in Q3
        - Hiring plan: 50 engineers, 20 sales reps
        """,
        doc_name="strategy_2025.txt",
    )

    AGENTIC_PROMPT = """You are a business analyst assistant.
Use search_knowledge_base to find relevant data before answering.
For complex questions, search multiple times with different queries to gather all needed info.
Always cite which document your data comes from."""

    agent = ReActAgent(llm=llm, tools=tools, system_prompt=AGENTIC_PROMPT, max_iterations=8)

    # Multi-hop question — needs info from both documents
    result = agent.run(
        "Compare our 2024 actual revenue with our 2025 target. "
        "What growth rate are we targeting and is it realistic given our trajectory?"
    )


# ── 3. RAG + Reflection (high-accuracy answers) ───────────────────────────────

def example_rag_with_reflection():
    """
    For use cases where accuracy is critical (legal, medical, financial).
    Critic verifies the answer is grounded in the retrieved documents.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: RAG + Reflection (verified answers)")
    print("="*60)

    cfg = Config()
    llm = cfg.build_llm()

    store = VectorStore(persist_dir=".agent_memory/rag_reflect")
    tools = ToolRegistry()
    register_builtin_tools(tools)
    register_rag_tools(tools, store)

    store.ingest_text(
        text="""
        LOAN AGREEMENT TERMS
        Interest rate: 8.5% per annum, fixed for 5 years
        Loan amount: $500,000
        Repayment period: 20 years
        Early repayment penalty: 2% of outstanding balance if repaid within first 3 years
        Default clause: 3 consecutive missed payments triggers default proceedings
        Collateral: Primary residence at 123 Main Street
        """,
        doc_name="loan_agreement.txt",
    )

    RAG_LEGAL_PROMPT = """You are a document analysis assistant.
ALWAYS search the knowledge base first.
Only state facts that are explicitly in the documents — never infer or assume.
Quote directly from the source when possible."""

    agent = ReActAgent(llm=llm, tools=tools, system_prompt=RAG_LEGAL_PROMPT)
    reflector = Reflector(llm, max_retries=1)

    task = "What is the early repayment penalty and under what conditions does default occur?"
    initial = agent.run(task)
    final = reflector.reflect_and_refine(agent, task, initial, verbose=True)
    print(f"\nVerified answer (score={final.score}/10):\n{final.final_answer}")


# ── How to ingest different source types ─────────────────────────────────────

def show_ingestion_options():
    print("""
Ingestion options:

    store = VectorStore()

    # Text string
    store.ingest_text("your content here", doc_name="my_doc")

    # Local file (.txt, .md, .py, .pdf)
    store.ingest_file("report.pdf")
    store.ingest_file("notes.md")
    store.ingest_file("code.py")

    # URL (fetches and parses the page)
    store.ingest_url("https://example.com/article")

    # Agent can also ingest at runtime:
    agent.run("Ingest the file at /path/to/report.pdf into the knowledge base")
    agent.run("Ingest https://example.com/docs into the knowledge base")
""")


if __name__ == "__main__":
    show_ingestion_options()
    example_simple_rag()
    # example_agentic_rag()
    # example_rag_with_reflection()
