# Agentic AI Boilerplate — Architecture & Flow

## 1. Core System Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR APPLICATION                            │
│                   (main.py / app.py / your script)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │      Config         │  ← env vars / .env
                    │  (provider, model,  │
                    │   keys, limits)     │
                    └──────────┬──────────┘
                               │ builds
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐    ┌────────▼────────┐   ┌──────▼──────┐
   │  LLM Layer  │    │  Tool Registry  │   │   Memory    │
   │  (any LLM)  │    │  (any tools)    │   │  (context)  │
   └──────┬──────┘    └────────┬────────┘   └──────┬──────┘
          │                    │                    │
          └────────────────────▼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Reasoning Engine   │  ← CORE LOOP
                    │  (ReAct / Planner)  │
                    │  (react.py)         │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐    ┌────────▼────────┐   ┌──────▼──────┐
   │  Planner    │    │  Router         │   │  Reflector  │
   │ (multi-step)│    │ (semantic route)│   │ (critique & │
   │  breakdown) │    │   / Orchestrate │   │  improve)   │
   └─────────────┘    └─────────────────┘   └─────────────┘
```

---

## 2. ReAct Loop (Core Pattern)

```
User Task
    │
    ▼
┌───────────────────────────────────────┐
│              ReAct Loop               │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │  1. THINK                       │  │
│  │     LLM reads: system prompt    │  │
│  │     + conversation history      │  │
│  │     + available tools           │  │
│  │     → decides what to do next   │  │
│  └──────────────┬──────────────────┘  │
│                 │                     │
│         has tool calls?               │
│          ┌──────┴──────┐              │
│         YES            NO             │
│          │              │             │
│  ┌───────▼────────┐     │             │
│  │  2. ACT        │     │             │
│  │  Execute tool  │     │             │
│  │  (shell, http, │     │             │
│  │   custom fn)   │     │             │
│  └───────┬────────┘     │             │
│          │              │             │
│  ┌───────▼────────┐     │             │
│  │  3. OBSERVE    │     │             │
│  │  Add result to │     │             │
│  │  conversation  │     │             │
│  └───────┬────────┘     │             │
│          │              │             │
│    loop back to         │             │
│       THINK             │             │
│                  ┌──────▼──────┐      │
│                  │ FINAL ANSWER│      │
│                  └─────────────┘      │
└───────────────────────────────────────┘
```

---

## 3. LLM Provider Layer

```
                    BaseLLM (abstract)
                    chat(messages, tools) → LLMResponse
                         │
          ┌──────────────┼──────────────┐
          │                             │
   OpenAILLM                    AnthropicLLM
   (openai_llm.py)              (anthropic_llm.py)
          │
   Works with ALL OpenAI-compatible APIs:
   ┌──────────────────────────────────┐
   │  OpenAI      gpt-4o, gpt-4o-mini │
   │  Groq        llama-3.1, gemma    │
   │  Ollama      llama3.2, mistral   │
   │  DeepSeek    deepseek-chat       │
   │  Together    mixtral, llama      │
   │  LM Studio   any local model     │
   │  Azure       gpt-4 via Azure     │
   └──────────────────────────────────┘

   AnthropicLLM works with:
   ┌──────────────────────────────────┐
   │  claude-haiku-4-5-20251001       │  ← fast, cheap
   │  claude-sonnet-4-6               │  ← balanced
   │  claude-opus-4-6                 │  ← most capable
   └──────────────────────────────────┘
```

---

## 4. Memory Architecture

```
┌─────────────────────────────────────────────────┐
│                  MEMORY SYSTEM                  │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │  Short-Term Memory (short_term.py)        │   │
│  │  Sliding window of last N messages        │   │
│  │  Lives in RAM — cleared on reset()        │   │
│  │                                           │   │
│  │  [system] [user] [assistant] [tool_result]│   │
│  │  [user]   [assistant] [tool_result] ...   │   │
│  │  ← keeps last 40 messages by default →   │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │  Long-Term Memory (long_term.py)          │   │
│  │  Persisted to disk (.agent_memory/)       │   │
│  │                                           │   │
│  │  KVStore     — key/value facts            │   │
│  │  kv.json       user prefs, known state    │   │
│  │                                           │   │
│  │  EpisodicStore — timestamped event log    │   │
│  │  episodes.jsonl  decisions, outcomes      │   │
│  │               keyword search built-in     │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```
---
## 5. Agentic Design Patterns — When to Use Each

```
┌───────────────────────────────────────────────────────────────────────┐
│  PATTERN          WHEN TO USE                  FILE                   │
├───────────────────────────────────────────────────────────────────────┤
│  ReAct            Always — core loop           agent/react.py         │
│  (Reason+Act)     Single task, iterative                              │
│                   "Just answer the question"                          │
│                                                                        │
│  Planner          Complex multi-step tasks     agent/planner.py       │
│  (Plan+Execute)   "Do X then Y then Z"                                │
│                   Long-horizon workflows                              │
│                                                                        │
│  Router           Choose different tools      agent/router.py        │
│  (Routing)        based on semantic content    SmartAgent             │
│                   Rule-based or LLM-based routing                     │
│                                                                        │
│  Reflection       High-stakes decisions        agent/reflection.py    │
│  (Critique+Retry) Quality matters > speed                             │
│                   Financial, medical, legal                           │
│                                                                        │
│  Orchestrator     Multiple specialized agents  agent/orchestrator.py  │
│  (Multi-Agent)    Different tools per domain                          │
│                   Parallel task execution                             │
│                                                                        │
│  Memory           Anything stateful            memory/long_term.py    │
│  (KV+Episodic)    User context, history                               │
│                   Persistent facts & events                           │
└───────────────────────────────────────────────────────────────────────┘
```
```
---

## 6. How to Adapt for ANY Use Case

### Example: Recommendation System (e.g. product / content / stock recs)

```
Step 1 — Add your domain tools  (tools/recommendations.py)
─────────────────────────────────────────────────────────
def register_recommendation_tools(registry):

    @registry.tool("Fetch user purchase history", ...)
    def get_user_history(user_id: str) -> str: ...

    @registry.tool("Search product catalog", ...)
    def search_products(query: str, filters: dict) -> str: ...

    @registry.tool("Get product details", ...)
    def get_product(product_id: str) -> str: ...

    @registry.tool("Get similar users (collaborative filtering)", ...)
    def get_similar_users(user_id: str, top_k: int) -> str: ...

    @registry.tool("Score a product for a user", ...)
    def score_product(user_id: str, product_id: str) -> str: ...

    @registry.tool("Store recommendation in DB", ...)
    def save_recommendation(user_id: str, recs: list) -> str: ...


Step 2 — Write a domain system prompt
──────────────────────────────────────
RECOMMENDATION_PROMPT = """
You are a personalised recommendation agent.
For each user:
1. Fetch their history and preferences
2. Search the catalog matching their taste
3. Score top candidates
4. Check for diversity (avoid repetition)
5. Return top-5 with reasoning
Never recommend out-of-stock items.
"""

Step 3 — Wire up the agent
──────────────────────────
cfg = Config()
llm = cfg.build_llm()
tools = ToolRegistry()
register_builtin_tools(tools)
register_recommendation_tools(tools)

agent = ReActAgent(
    llm=llm,
    tools=tools,
    system_prompt=RECOMMENDATION_PROMPT,
)

result = agent.run("Recommend products for user_id=U123")
print(result.answer)


Step 4 — Add Reflection for quality check (optional)
────────────────────────────────────────────────────
reflector = Reflector(llm, max_retries=1)
initial = agent.run("Recommend products for user_id=U123")
final = reflector.reflect_and_refine(agent, task, initial)


Step 5 — Scale with Orchestrator (optional)
────────────────────────────────────────────
profile_agent  → builds user profile
catalog_agent  → searches + scores products
ranking_agent  → ranks and diversifies
delivery_agent → saves to DB / sends notification

orchestrator.run_pipeline([
    "Build profile for user U123",
    "Find candidate products for this user profile",
    "Rank and diversify the candidates",
    "Save top-5 recommendations for user U123",
])
```

---

## 7. Full File Map

```
my_agent/

├── config.py             ← Single config object, reads env vars
│
├── llm/
│   ├── base.py           ← BaseLLM, LLMResponse, Message, ToolCall
│   ├── openai_llm.py     ← OpenAI / Groq / Ollama / DeepSeek / Together / Azure
│   └── anthropic_llm.py  ← Anthropic Claude models
│
├── tools/
│   ├── registry.py       ← ToolRegistry + @registry.tool decorator
│   ├── builtin.py        ← shell, file I/O, HTTP requests, Python REPL
│   ├── rag.py            ← VectorStore, embeddings, semantic search
│   ├── arxiv.py          ← arXiv API, paper fetching (example tool)
│   ├── nl_to_sql.py      ← Trino SQL generation (example tool)
│   ├── query_history.py  ← NL→SQL query caching (example tool)
│   └── table_relationships.py ← FK graph for join discovery (example tool)
│
├── memory/
│   ├── short_term.py     ← Sliding-window conversation history (max N msgs)
│   └── long_term.py      ← KVStore (facts) + EpisodicStore (event log)
│
├── agent/
│   ├── react.py          ← ReAct loop — CORE REASONING ENGINE
│   ├── planner.py        ← Plan → subtask decomposition + execution
│   ├── reflection.py     ← Critic → refine loop (critique + improve)
│   ├── orchestrator.py   ← Multi-agent coordination + parallel execution
│   └── router.py         ← Semantic routing (rule-based and LLM-based)
│
├── observability/
│   └── tracer.py         ← Langfuse v3 integration (SPAN/GENERATION/TOOL)
│
├── examples/
│   ├── rag_example.py              ← Simple RAG, Agentic RAG, Multi-doc RAG
│   ├── trading_agent_example.py    ← Full trading use case
│   └── nl_to_sql_example.py        ← NLP-to-SQL with 500+ tables
│
├── app.py                ← Streamlit chatbot UI (optional deployment)
├── main.py               ← CLI entry point / interactive REPL
├── requirements.txt
└── .env.example
```

### Quick Start (3 steps)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# or: export LLM_PROVIDER=groq && export GROQ_API_KEY=gsk_...

# 3. Run
python main.py                          # interactive REPL
python main.py "what files are here?"   # single task
```

---

## 8. All Use Cases This Boilerplate Can Power

> The only thing that changes per use case is **what tools you register** and **what the system prompt says**.
> The agent loop, memory, reflection, and orchestration stay exactly the same.

---

### Finance & Trading

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| Trading agent (perps/spot) | Exchange APIs, on-chain data | Orchestrator + Reflection |
| Portfolio rebalancer | Price feeds, order APIs | ReAct + Risk check |
| Earnings call analyser | PDF parser, SEC filings API | ReAct + Planner |
| Fraud detection agent | Transaction DB, rule engine | ReAct + Reflection |
| Credit risk scorer | Financial data APIs | ReAct + Reflection |

---

### Data & Analytics

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| Auto data analyst | SQL tool, pandas REPL, chart gen | ReAct |
| Report generator | DB queries, PDF/Excel writer | Planner → Execute |
| Anomaly detector | Metrics API, alerting tool | Continuous ReAct loop |
| Business intelligence bot | Warehouse queries, dashboards | ReAct + Memory |

---

### Developer Tools

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| Code review agent | Git diff, linter, test runner | ReAct + Reflection |
| Bug fixer | Shell, file read/write, test tool | ReAct |
| CI/CD debugger | Log fetcher, deploy tools | ReAct |
| Documentation writer | File reader, markdown writer | Planner → Execute |
| Dependency auditor | Package scanners, CVE APIs | ReAct |

---

### Customer & Product

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| Recommendation engine | User history DB, catalog search | Orchestrator |
| Customer support agent | CRM API, knowledge base search | ReAct + Memory |
| Lead scoring agent | CRM data, enrichment APIs | ReAct + Reflection |
| Onboarding agent | Form tools, email sender | Planner → Execute |
| Churn predictor | User events DB, model scoring | ReAct |

---

### Research & Knowledge

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| Research assistant | Web search, PDF reader, note saver | ReAct + Long-term memory |
| Competitive intel agent | Web scraper, news API | Planner + Orchestrator |
| Patent analyser | USPTO API, PDF parser | ReAct + Reflection |
| Literature reviewer | ArXiv API, summariser | Planner → Execute |
| Legal document reviewer | PDF reader, clause extractor | ReAct + Reflection |

---

### Operations & Infrastructure

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| DevOps agent | Kubernetes API, cloud SDK | ReAct |
| Incident responder | Log tools, alerting, runbooks | ReAct + Reflection |
| Cost optimiser | Cloud billing API, infra tools | Planner |
| Security scanner | CVE checker, network tools | ReAct |

---

### Content & Media

| Use Case | Tools to Add | Pattern |
|----------|-------------|---------|
| SEO content agent | Keyword API, web scraper, writer | Planner → Execute |
| Social media scheduler | Platform APIs, content DB | ReAct |
| Newsletter generator | RSS feeds, summariser, emailer | Planner |
| Video script writer | Trend API, outline → script | Planner + Reflection |

---

### The Formula (same every time)

```
1. Add tools/your_domain.py
      → functions that talk to your APIs / DBs / services

2. Write a system_prompt
      → defines agent's role + rules + workflow steps

3. Pick a pattern:
      simple task       →  ReActAgent
      complex task      →  Planner + ReActAgent
      high stakes       →  ReActAgent + Reflector
      multiple domains  →  Orchestrator
      stateful / memory →  + Long-term Memory
```

---

## 10. Optional Components

### Observability — Langfuse v3 Integration

**When to use** — Track agent performance, token usage, costs, and debug complex reasoning chains

**Location** — `observability/tracer.py`

**Features:**
- Wraps `agent.run()` to emit SPAN/GENERATION/TOOL observations
- Tracks: prompt tokens, completion tokens, cost (USD), latency (ms)
- Session-level tracking for user cohorts
- Cost estimation for 20+ models (Claude, GPT, Groq, local models)

```python
from observability.tracer import AgentTracer

tracer = AgentTracer(session_id="user_123", model="claude-haiku-4-5-20251001")
result = tracer.trace_run(agent, "your task here")  # emits to Langfuse
print(result.summary())  # "trace=abc123 | cost=$0.00042 | latency=1250ms"
```

### Optional Tools — Examples

**RAG (tools/rag.py):**
- VectorStore with pluggable embeddings (SentenceTransformer, OpenAI, BM25)
- Semantic search + deduplication by doc_id
- Chunking with configurable overlap

**ArXiv (tools/arxiv.py):**
- Search papers by date range and keywords
- Fetch full paper text (cache-first)
- Batch fetching with ThreadPoolExecutor

**NLP-to-SQL (tools/nl_to_sql.py):**
- Generate SQL from natural language
- Schema RAG for 500+ table discovery
- Join path inference via FK graph
- Query history for few-shot learning

---

## 11. Getting Started

### Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...  # or choose another provider
```

### 1-Minute Example

```python
from config import Config
from tools import ToolRegistry, register_builtin_tools
from agent import ReActAgent

cfg = Config()
llm = cfg.build_llm()
tools = ToolRegistry()
register_builtin_tools(tools)

agent = ReActAgent(llm=llm, tools=tools)
result = agent.run("What files are in the current directory?")
print(result.answer)
```

---

## 12. Entry Points

- **CLI/REPL** — `main.py` — Interactive loop with `python main.py`
- **Streamlit UI** — `app.py` — Visual chatbot interface, run `streamlit run app.py`
- **Direct Python** — Import and call `ReActAgent.run()` or `.chat()`
- **Examples** — `examples/` — Pre-built use cases (RAG, trading, NLP-to-SQL)