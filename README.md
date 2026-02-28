# Agentic AI Framework

<div align="center">
  <h3>A Production-Grade, Provider-Agnostic Agentic AI Framework</h3>
  <p>From ReAct loops to multi-agent orchestration — built from scratch, no LangChain required</p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/LLMs-OpenAI%20|%20Anthropic%20|%20Groq%20|%20Ollama-green.svg" alt="LLM Providers">
  <img src="https://img.shields.io/badge/Patterns-5%20Agent%20Patterns-orange.svg" alt="Agent Patterns">
  <img src="https://img.shields.io/badge/Observability-Langfuse%20v3-purple.svg" alt="Observability">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg" alt="Status">
</p>

---

## What Is This?

A **custom agentic AI framework** built entirely from scratch — no LangChain, no LlamaIndex, no CrewAI. Every component (LLM adapters, tool registry, memory, agent loops, RAG pipeline, observability) is hand-written Python, giving you full control and understanding of every layer.

The framework powers a **Streamlit chatbot** that can reason, use tools, search academic papers, generate code, create its own tools at runtime, and answer questions over ingested documents.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Chatbot UI                         │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐            │
│  │ Provider  │  │ Persona  │  │  arXiv  │  │ Dynamic  │            │
│  │ Selector  │  │ Selector │  │ Sidebar │  │  Tools   │            │
│  └────┬─────┘  └────┬─────┘  └────┬────┘  └────┬─────┘            │
│       └──────────────┴─────────────┴────────────┘                  │
│                           │                                         │
│              Background Thread + Stop Button                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Layer (5 Patterns)                       │
│                                                                     │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────────┐           │
│  │  ReAct  │  │ Planner │  │Reflector │  │Orchestrator│           │
│  │  Agent  │  │+Executor│  │ (Critic) │  │(Multi-Agent)│          │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └─────┬──────┘           │
│       └─────────────┴───────────┴───────────────┘                  │
│                           │                                         │
│              SmartAgent / TaskRouter                                │
│         (auto-selects pattern per task)                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
│    LLM Layer     │ │  Tools   │ │     Memory       │
│                  │ │          │ │                   │
│ ┌──────────────┐ │ │BuiltIn 6│ │ ShortTermMemory   │
│ │   OpenAI     │ │ │arXiv   4│ │ (sliding window)  │
│ │ (+ Groq,     │ │ │RAG     4│ │                   │
│ │  Ollama,     │ │ │Dynamic 3│ │ LongTermMemory    │
│ │  Together,   │ │ │NL-SQL  7│ │ (KV + Episodic)   │
│ │  DeepSeek)   │ │ │         │ │                   │
│ ├──────────────┤ │ │ToolRouter│ │                   │
│ │  Anthropic   │ │ │(≤10/call)│ │                   │
│ └──────────────┘ │ └──────────┘ └──────────────────┘
└──────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Observability (Langfuse v3)                      │
│  Traces → Spans → LLM calls, tool calls, cost, latency, tokens     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Agent Patterns

The framework implements **5 distinct agentic patterns**, each suited to different task types. The `SmartAgent` auto-selects the best pattern per query.

### ReAct Agent (Reasoning + Acting)

The core loop. The agent thinks, acts (calls a tool), observes the result, and repeats until it has an answer.

```
┌──────────────────────────────────────────────────────┐
│                   ReAct Loop                         │
│                                                      │
│   ┌─────────┐     ┌─────────┐     ┌──────────┐     │
│   │  THINK  │────▶│   ACT   │────▶│ OBSERVE  │     │
│   │(reason) │     │(tool    │     │(read     │     │
│   │         │     │ call)   │     │ result)  │     │
│   └─────────┘     └─────────┘     └────┬─────┘     │
│       ▲                                │            │
│       └────────────────────────────────┘            │
│                                                      │
│   Guards:                                            │
│   • max_iterations (default 15)                      │
│   • stop_event (threading.Event for cancellation)    │
│   • error_counts (abort after 2 identical failures)  │
│   • tool deduplication (≤5 per iteration)            │
└──────────────────────────────────────────────────────┘
```

**Best for:** Most tasks — research, Q&A, tool-use, code generation.

### Planner + Executor (Task Decomposition)

Breaks complex goals into ordered subtasks, then executes each with a ReAct agent.

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  User Goal ──▶ Planner LLM ──▶ [Step 1, Step 2, Step 3, ...] │
│                                       │                        │
│                                       ▼                        │
│                    ┌──────────────────────────────────┐        │
│                    │        PlanExecutor              │        │
│                    │                                  │        │
│                    │  Step 1 ──▶ ReAct ──▶ Result 1   │        │
│                    │       (context from prev steps)  │        │
│                    │  Step 2 ──▶ ReAct ──▶ Result 2   │        │
│                    │       (context: Result 1)        │        │
│                    │  Step 3 ──▶ ReAct ──▶ Result 3   │        │
│                    │       (context: Result 1+2)      │        │
│                    │                                  │        │
│                    │  Memory reset between steps      │        │
│                    └──────────────────────────────────┘        │
│                                       │                        │
│                                       ▼                        │
│                              Final Answer                      │
└────────────────────────────────────────────────────────────────┘
```

**Best for:** Multi-step research, "compare X and Y then recommend", build-then-test workflows.

### Reflector (Self-Evaluation)

After the agent produces an answer, a separate critic LLM evaluates it on 4 dimensions. If it fails, the agent retries with feedback.

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  Query ──▶ ReAct Agent ──▶ Draft Answer                │
│                                 │                      │
│                                 ▼                      │
│                          ┌────────────┐                │
│                          │  Critic    │                │
│                          │  LLM      │                │
│                          │           │                │
│                          │ Score 0-10│                │
│                          │ on each:  │                │
│                          │ • Complete│                │
│                          │ • Accurate│                │
│                          │ • Safe    │                │
│                          │ • Clear   │                │
│                          └─────┬─────┘                │
│                                │                      │
│                    ┌───────────┴──────────┐            │
│                    │                      │            │
│               PASS (≥7)            FAIL (<7)           │
│                    │                      │            │
│                    ▼                      ▼            │
│              Return Answer      Inject critique        │
│                                 + retry (max 2)        │
└────────────────────────────────────────────────────────┘
```

**Best for:** High-stakes answers, medical/legal/financial queries, anything needing quality assurance.

### Orchestrator (Multi-Agent)

Routes tasks to specialized sub-agents. Supports parallel execution and pipeline chaining.

```
┌────────────────────────────────────────────────────────────────┐
│                        Orchestrator                            │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │                  LLM Router                          │     │
│  │  "Which sub-agent should handle this task?"          │     │
│  └───────────┬──────────────┬───────────────┬───────────┘     │
│              │              │               │                  │
│              ▼              ▼               ▼                  │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│     │ Data Agent │  │ Analysis   │  │ Execution  │            │
│     │ (fetch,    │  │ Agent      │  │ Agent      │            │
│     │  scrape)   │  │ (compute,  │  │ (write,    │            │
│     │            │  │  reason)   │  │  act)      │            │
│     └────────────┘  └────────────┘  └────────────┘            │
│                                                                │
│  Execution modes:                                              │
│  • run()          — single task, auto-routed                   │
│  • run_parallel() — multiple tasks, ThreadPoolExecutor         │
│  • run_pipeline() — sequential with context chaining           │
└────────────────────────────────────────────────────────────────┘
```

**Best for:** Complex multi-domain tasks, "fetch data then analyze then execute".

### SmartAgent (Auto-Pattern Selection)

Automatically picks the best pattern for each task using rule-based or LLM-based routing.

```
┌────────────────────────────────────────────────────────┐
│                    SmartAgent                           │
│                                                        │
│  Query ──▶ TaskRouter                                  │
│                │                                       │
│       ┌───────┴───────┐                                │
│       │               │                                │
│  rule-based       LLM-based                            │
│  (free, instant)  (smarter)                            │
│       │               │                                │
│       └───────┬───────┘                                │
│               │                                        │
│        Classification:                                 │
│        ┌──────┼──────┬──────────┐                      │
│        ▼      ▼      ▼          ▼                      │
│     react  plan   reflect  orchestrate                 │
│                                                        │
│  Optional: stack reflection on top of any pattern      │
└────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline

Full Retrieval-Augmented Generation with multiple embedding backends and chunking strategies.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ingestion Pipeline                       │
│                                                                 │
│  Document ──▶ SemanticChunker ──▶ Embedder ──▶ VectorStore     │
│  (.txt, .md,    │                    │            (pickle)      │
│   .py, .pdf,    │                    │                          │
│   URL, text)    │                    │                          │
│                 ▼                    ▼                          │
│          ┌────────────┐     ┌──────────────┐                   │
│          │ Rust-based │     │ 3 backends:  │                   │
│          │ splitter   │     │ • OpenAI     │                   │
│          │ (primary)  │     │   (1536-dim) │                   │
│          │            │     │ • MiniLM     │                   │
│          │ Python     │     │   (384-dim)  │                   │
│          │ regex      │     │ • BM25       │                   │
│          │ (fallback) │     │   (no embed) │                   │
│          └────────────┘     └──────────────┘                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Retrieval Pipeline                       │
│                                                                 │
│  User Query                                                     │
│       │                                                         │
│       ├──▶ Cosine Similarity (if neural embeddings available)   │
│       │                                                         │
│       └──▶ BM25 Scoring (k1=1.5, b=0.75, zero-dependency)      │
│                                                                 │
│       Deduplication: one best chunk per document                │
│       │                                                         │
│       ▼                                                         │
│  Top-K Results ──▶ Agent Context ──▶ LLM generates answer      │
└─────────────────────────────────────────────────────────────────┘
```

---

## ArXiv Integration

Automated academic paper fetching, caching, and indexing into the RAG knowledge base.

```
┌────────────────────────────────────────────────────────────────────┐
│                        arXiv Pipeline                              │
│                                                                    │
│  search_arxiv(topic)                                               │
│       │                                                            │
│       ▼                                                            │
│  arXiv API ──▶ Parse XML ──▶ Paper Metadata                       │
│                                (title, authors, abstract, ID)      │
│                                                                    │
│  fetch_arxiv_paper(paper_id)                                       │
│       │                                                            │
│       ├──▶ PaperCache hit? ──▶ Return cached                      │
│       │         │                                                  │
│       │         ▼ (miss)                                           │
│       ├──▶ Try HTML (cleaner) ──▶ Strip tags ──▶ Text              │
│       │         │                                                  │
│       │         ▼ (fail)                                           │
│       └──▶ Try PDF ──▶ pypdf extract ──▶ Text                     │
│                                            │                       │
│                                            ▼                       │
│                              SemanticChunker ──▶ VectorStore       │
│                                                                    │
│  fetch_arxiv_papers_batch(ids)                                     │
│       └──▶ ThreadPoolExecutor(max_workers=5), up to 10 papers      │
│                                                                    │
│  Paper ID validation: ^\d{4}\.\d{4,5}(v\d+)?$ + legacy format     │
│  PaperCache: disk-based, 24h TTL, JSON in .arxiv_cache/           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Tool System

### Tool Router (Query-Time Selection)

Keeps tool count under model limits (Groq caps at ~10 tools). Re-evaluated every ReAct iteration.

```
┌──────────────────────────────────────────────────────────────┐
│                        ToolRouter                            │
│                                                              │
│  User Query: "search arxiv for transformers"                 │
│       │                                                      │
│       ▼                                                      │
│  Keyword Matching                                            │
│  ┌────────────┬───────────────────────────────────┐          │
│  │ Group      │ Keywords                          │          │
│  ├────────────┼───────────────────────────────────┤          │
│  │ core       │ (always included)                 │          │
│  │ file       │ file, read, write, search, save   │          │
│  │ shell      │ shell, run, execute, command      │          │
│  │ web        │ http, url, request, fetch, api    │          │
│  │ arxiv      │ arxiv, paper, research, academic  │          │
│  │ rag        │ knowledge, document, ingest, rag  │          │
│  │ dynamic    │ create tool, dynamic, custom      │          │
│  └────────────┴───────────────────────────────────┘          │
│       │                                                      │
│       ▼                                                      │
│  Selected: [python_repl, search_arxiv, fetch_arxiv_paper,    │
│             list_arxiv_papers] (≤10 total)                   │
│                                                              │
│  + Custom dynamic tools always included                      │
│  + Re-evaluated every iteration (new tools visible mid-run)  │
└──────────────────────────────────────────────────────────────┘
```

### Dynamic Tool Synthesis

The agent can create, save, and load new Python tools at runtime.

```
┌────────────────────────────────────────────────────────────────┐
│                    Dynamic Tool Lifecycle                       │
│                                                                │
│  Agent: "I need a tool to convert Celsius to Fahrenheit"       │
│       │                                                        │
│       ▼                                                        │
│  create_tool(                                                  │
│    name="celsius_to_fahrenheit",                               │
│    description="Convert temperature",                          │
│    code="def celsius_to_fahrenheit(temp): return temp*9/5+32"  │
│  )                                                             │
│       │                                                        │
│       ├──▶ exec() the code ──▶ Register in ToolRegistry        │
│       │                                                        │
│       └──▶ Save to .agent_tools/tools.json (persistent)        │
│                                                                │
│  On next startup:                                              │
│       load_saved_tools() ──▶ Restore all custom tools          │
│                                                                │
│  Context-efficient: only name + description + schema           │
│  sent to LLM (50-120 tokens per tool, not full source code)    │
└────────────────────────────────────────────────────────────────┘
```

---

## LLM Provider Support

Single codebase, multiple providers — swap models with one config change.

```
┌──────────────────────────────────────────────────────────────────┐
│                       BaseLLM Interface                          │
│              chat(messages, tools?) → LLMResponse                │
│              stream_chat(messages, tools?) → Iterator            │
│                          │                                       │
│              ┌───────────┴───────────┐                           │
│              ▼                       ▼                           │
│     ┌──────────────┐       ┌──────────────┐                     │
│     │  OpenAILLM   │       │AnthropicLLM  │                     │
│     │              │       │              │                     │
│     │ Works with:  │       │ Works with:  │                     │
│     │ • OpenAI     │       │ • Claude 3.5 │                     │
│     │ • Groq       │       │ • Claude 3   │                     │
│     │ • Ollama     │       │ • Claude 4   │                     │
│     │ • Together   │       │              │                     │
│     │ • DeepSeek   │       └──────────────┘                     │
│     │ • Mistral    │                                             │
│     │ • LM Studio  │       Special Features:                     │
│     │ • Anyscale   │       • Groq malformed tool call recovery   │
│     └──────────────┘       • Streaming support for both          │
│                            • Auto tool schema conversion         │
└──────────────────────────────────────────────────────────────────┘
```

---

## NLP-to-SQL Pipeline

Full natural-language-to-SQL system with Trino, schema RAG, join discovery, and Spider benchmark evaluation.

```
┌─────────────────────────────────────────────────────────────────┐
│                     NLP-to-SQL Pipeline                         │
│                                                                 │
│  User: "Show me top customers by revenue last month"            │
│       │                                                         │
│       ├──▶ SchemaIndexer ──▶ VectorStore search ──▶ Relevant    │
│       │    (crawls Trino        tables + columns)               │
│       │                                                         │
│       ├──▶ RelationshipRegistry ──▶ BFS join path discovery     │
│       │    (infers FK from *_id        max 4 hops)              │
│       │         column patterns)                                │
│       │                                                         │
│       ├──▶ QueryHistory ──▶ Similar past queries (few-shot)     │
│       │    (JSONL store with       keyword similarity)           │
│       │                                                         │
│       ▼                                                         │
│  ReAct Agent ──▶ SQL Generation ──▶ Execute on Trino            │
│       │                                │                        │
│       │                                ▼                        │
│       │                          ASCII table output             │
│       │                          (500-row hard cap)             │
│       │                                                         │
│  Optional: + Reflection (verify SQL correctness)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Spider Benchmark Evaluation                       │
│                                                                 │
│  Generate ──▶ 5-Layer Gate ──▶ Retry (if failed)               │
│  (GPT-4o,      │                                                │
│   CoT,         ├─ 1. Syntax (sqlglot)                          │
│   self-        ├─ 2. Schema compliance                          │
│   consistency) ├─ 3. Execution test (SQLite)                    │
│                ├─ 4. LLM-as-Judge (4 dimensions, 1-5)          │
│                └─ 5. Execution equivalence vs gold SQL          │
│                                                                 │
│  Metrics: Exact Match, Execution Accuracy, Pass Rate, Score     │
│  Export: CSV, JSON, Langfuse experiments                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Observability

End-to-end tracing via Langfuse v3.

```
┌────────────────────────────────────────────────────────────┐
│                   AgentTracer (Langfuse v3)                │
│                                                            │
│  Trace                                                     │
│  ├── Span: LLM Call #1                                     │
│  │   ├── model: llama-3.3-70b-versatile                    │
│  │   ├── input tokens: 1,240                               │
│  │   ├── output tokens: 380                                │
│  │   ├── latency: 2.3s                                     │
│  │   └── cost: $0.0012                                     │
│  ├── Span: Tool Call — search_arxiv                        │
│  │   ├── input: {"topic": "transformers"}                  │
│  │   ├── output: [5 papers found]                          │
│  │   └── latency: 1.1s                                     │
│  ├── Span: LLM Call #2                                     │
│  │   └── ...                                               │
│  └── Metadata                                              │
│      ├── session_id, trace_id                              │
│      ├── total_llm_calls: 3                                │
│      ├── total_tool_calls: 2                               │
│      ├── total_tokens: 4,200                               │
│      └── total_cost: $0.0038                               │
│                                                            │
│  Cost table: Claude, GPT-4o, Llama 3.1/3.3 per 1M tokens  │
│  Non-blocking flush: daemon thread                         │
└────────────────────────────────────────────────────────────┘
```

---

## Streamlit Chatbot UI

Feature-rich chat interface with provider selection, persona system, and live tool visibility.

```
┌──────────────────────────────────────────────────────────────────┐
│  Sidebar                          │  Chat Area                   │
│  ┌────────────────────────┐       │  ┌────────────────────────┐  │
│  │ Provider: [Groq    ▼]  │       │  │ 🤖 Assistant           │  │
│  │ Model: [llama-3.3 ▼]  │       │  │ Here are the latest    │  │
│  │ API Key: [••••••••]    │       │  │ papers on transformers │  │
│  ├────────────────────────┤       │  │                        │  │
│  │ Persona:               │       │  │ ▶ Tool: search_arxiv   │  │
│  │ ○ General Assistant    │       │  │   {topic: "transform…"}│  │
│  │ ● Data Analyst         │       │  │                        │  │
│  │ ○ Code Helper          │       │  │ ▶ Tool: fetch_arxiv    │  │
│  │ ○ Customer Support     │       │  │   {id: "2401.12345"}   │  │
│  │ ○ Custom               │       │  │                        │  │
│  ├────────────────────────┤       │  │ ── trace info ──       │  │
│  │ ☑ Enable Tools         │       │  │ LLM calls: 3           │  │
│  │ Max iterations: [15]   │       │  │ Tokens: 4,200          │  │
│  ├────────────────────────┤       │  │ Cost: $0.004           │  │
│  │ arXiv Papers           │       │  │ Latency: 5.2s          │  │
│  │ [Search papers...]     │       │  └────────────────────────┘  │
│  │ [Add by ID...]         │       │                              │
│  │ 📄 Paper 1             │       │  ┌────────────────────────┐  │
│  │ 📄 Paper 2             │       │  │ [Type a message... ]   │  │
│  ├────────────────────────┤       │  │          [⏹ Stop]      │  │
│  │ Dynamic Tools          │       │  └────────────────────────┘  │
│  │ 🔧 celsius_to_fahr [×] │       │                              │
│  ├────────────────────────┤       │  Arrow Up/Down: chat history │
│  │ Langfuse: ☑ Enabled    │       │  Background thread execution │
│  │ Stats: 3 turns, $0.01  │       │                              │
│  └────────────────────────┘       │                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
agentic-ai-apps/
├── my_agent/                          # Core framework (provider-agnostic)
│   ├── agent/
│   │   ├── react.py                   # ReAct loop (THINK → ACT → OBSERVE)
│   │   ├── planner.py                 # Task decomposition + sequential execution
│   │   ├── reflection.py              # Critic LLM for answer verification
│   │   ├── orchestrator.py            # Multi-agent routing + parallel execution
│   │   └── router.py                  # SmartAgent auto-pattern selection
│   ├── llm/
│   │   ├── base.py                    # BaseLLM, Message, ToolCall, LLMResponse
│   │   ├── openai_llm.py             # OpenAI-compatible (Groq, Ollama, etc.)
│   │   └── anthropic_llm.py          # Anthropic Claude
│   ├── tools/
│   │   ├── registry.py               # Tool registration + execution
│   │   ├── router.py                 # Query-time tool selection (≤10 per call)
│   │   ├── builtin.py                # shell, file, HTTP, python_repl
│   │   ├── arxiv.py                  # arXiv search, fetch, batch, cache
│   │   ├── rag.py                    # VectorStore, BM25, SemanticChunker
│   │   ├── dynamic.py                # Runtime tool creation + persistence
│   │   ├── nl_to_sql.py              # Trino connector, schema indexer
│   │   ├── query_history.py          # NL-to-SQL few-shot cache
│   │   └── table_relationships.py    # FK inference, BFS join paths
│   ├── memory/
│   │   ├── short_term.py             # Sliding window (smart trimming)
│   │   └── long_term.py              # KVStore + EpisodicStore
│   ├── observability/
│   │   └── tracer.py                 # Langfuse v3 tracing
│   ├── config.py                     # Centralized config from env vars
│   ├── main.py                       # CLI entry point (task or REPL)
│   ├── app.py                        # Simple Streamlit app
│   └── examples/
│       ├── trading_agent_example.py   # 8 stub tools, 4 pattern demos
│       ├── rag_example.py             # Simple, agentic, reflected RAG
│       ├── nl_to_sql_example.py       # Trino NL-to-SQL REPL
│       └── nlp_to_sql_spider/         # Spider benchmark evaluation
│           ├── generator.py           # GPT-4o with CoT + self-consistency
│           ├── gatekeeper.py          # 5-layer SQL validation
│           ├── evaluator.py           # Batch metrics + Langfuse experiments
│           ├── spider_loader.py       # HuggingFace dataset integration
│           ├── prompt_manager.py      # Langfuse prompt management
│           ├── server.py              # FastAPI REST API
│           └── cli.py                 # CLI: query, evaluate, REPL
│
├── chatbot_app/                       # Streamlit chatbot UI
│   ├── app.py                         # Main app (background thread runner)
│   ├── sidebar.py                     # Provider, persona, arXiv, tools sidebar
│   ├── ui.py                          # Chat rendering, error formatting
│   ├── agent_runner.py                # Agent construction + background workers
│   ├── state.py                       # Session state management
│   └── constants.py                   # Models, providers, defaults
│
├── comparison.md                      # Comparison with production RAG course
├── plan.md                            # Detailed enhancement roadmap
└── README.md                          # This file
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- An API key for at least one provider (Groq is free)

### Setup

```bash
# Clone the repo
git clone https://github.com/your-username/agentic-ai-apps.git
cd agentic-ai-apps

# Install agent framework dependencies
cd my_agent && pip install -r requirements.txt && cd ..

# Install chatbot app dependencies
cd chatbot_app && pip install -r requirements.txt && cd ..

# Set environment variables
export GROQ_API_KEY="your-key-here"          # Free at console.groq.com
# Optional:
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export LANGFUSE_PUBLIC_KEY="your-key-here"   # Optional: for tracing
export LANGFUSE_SECRET_KEY="your-key-here"
```

### Run the Chatbot

```bash
cd chatbot_app
streamlit run app.py
```

### Run the CLI

```bash
cd my_agent
python main.py "What are the latest papers on reasoning in LLMs?"
# or interactive mode:
python main.py
```

### Run Examples

```bash
# RAG example
python my_agent/examples/rag_example.py

# Trading agent example
python my_agent/examples/trading_agent_example.py

# NL-to-SQL example (requires Trino)
python my_agent/examples/nl_to_sql_example.py
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **No LangChain/LlamaIndex** | Full control over every layer. Easier to debug, extend, and understand. No hidden abstractions. |
| **BM25 over TF-IDF** | Term-frequency saturation + document-length normalization. Better retrieval at zero extra cost. |
| **Keyword ToolRouter over embeddings** | Instantaneous, deterministic, free. For 6 tool groups, keywords are precise enough. |
| **Background thread (not async)** | Streamlit is synchronous. Background thread + polling is the standard Streamlit pattern. |
| **`st.components.v1.html()` for JS** | `st.markdown(unsafe_allow_html=True)` doesn't execute `<script>` tags (HTML5 spec). Components create real iframes. |
| **Groq recovery over model switching** | The 400 error contains exactly what the model tried to call. Parsing is free. Agent never knows it happened. |
| **Per-iteration router evaluation** | Dynamic tools created mid-run are immediately visible to the agent. Costs microseconds. |
| **Sliding window memory** | Smart trim that never splits assistant+tool_results groups. Prevents orphaned tool messages causing API 400 errors. |

---

## Resilience Features

- **Groq malformed tool call recovery**: Regex parses `<function=name{args}</function>` from 400 errors, returns valid `ToolCall`
- **Retry-loop guard**: Aborts after 2 identical tool failures with injected "stop retrying" message
- **Tool deduplication**: Same (name, args) calls within one response capped at 5
- **Memory trimming**: Advances cut point to next `user` message, never splits tool result groups
- **Tool result truncation**: Large results capped at 3,000 chars before storing in memory
- **Date injection**: System prompt includes `date.today()` in ISO and compact formats
- **Stop button**: `threading.Event` checked at every iteration for graceful cancellation
- **Background execution**: Daemon threads with full traceback logging on error

---

## Status

🚧 **Active Development** - This project is under active experimentation. APIs and architectures may change as we explore what works best.