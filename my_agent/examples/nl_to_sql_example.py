"""
NLP-to-SQL on Trino — Full Production Setup
─────────────────────────────────────────────
Wires together all 4 layers:
  1. Schema RAG         — find relevant tables from 500+ using semantic search
  2. Join discovery     — know how tables connect (auto-inferred + manual)
  3. Query history      — reuse past correct queries as few-shot examples
  4. ReAct + Reflection — generate, execute, fix, and verify SQL

Run order:
  Step 1: index_everything()  ← run ONCE (or on schema change)
  Step 2: run_repl()          ← interactive NL→SQL shell

env vars:
    TRINO_HOST, TRINO_PORT, TRINO_USER, TRINO_CATALOG, TRINO_SCHEMA
    ANTHROPIC_API_KEY  (or LLM_PROVIDER + matching key)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from tools import ToolRegistry, register_builtin_tools
from tools.rag import VectorStore
from tools.nl_to_sql import TrinoConnector, SchemaIndexer, register_nl_to_sql_tools
from tools.query_history import QueryHistory, register_query_history_tools
from tools.table_relationships import RelationshipRegistry, register_relationship_tools
from agent import ReActAgent, Reflector


# ── Config ────────────────────────────────────────────────────────────────────

TRINO_HOST    = os.getenv("TRINO_HOST",    "trino.yourcompany.com")
TRINO_PORT    = int(os.getenv("TRINO_PORT", "443"))
TRINO_USER    = os.getenv("TRINO_USER",    "analyst")
TRINO_CATALOG = os.getenv("TRINO_CATALOG", "hive")
TRINO_SCHEMA  = os.getenv("TRINO_SCHEMA",  "analytics")

MEMORY_DIR = ".agent_memory/nl_to_sql"


# ── SQL Agent system prompt ───────────────────────────────────────────────────

SQL_SYSTEM_PROMPT = f"""You are an expert Trino SQL analyst for a large data warehouse with 500+ tables.

YOUR EXACT WORKFLOW for every question:
  1. get_similar_queries(question)      ← check if similar query was done before
  2. search_schema(question)            ← find relevant tables from the warehouse
  3. get_table_schema(table)            ← get exact column names for relevant tables
  4. find_join_path(t1, t2)             ← if joining tables, check how they connect
  5. get_sample_data(table)             ← if unsure about data format/values
  6. Write the SQL
  7. explain_sql(query)                 ← validate syntax (optional, for complex queries)
  8. execute_sql(query)                 ← run it
  9. If error → fix SQL and retry (max 3 retries)
  10. save_successful_query(q, sql)     ← save correct query to history

Trino SQL rules:
  - Always use fully qualified names: {TRINO_CATALOG}.schema.table
  - Use DATE_TRUNC('month', ts) for month grouping
  - Use CURRENT_DATE, CURRENT_TIMESTAMP for time references
  - Use TRY_CAST() to safely handle type conversions
  - Use APPROX_DISTINCT() for fast cardinality estimates on large tables
  - Use WITH (CTE) for multi-step logic — more readable than nested subqueries
  - Always add LIMIT unless user explicitly asks for all rows
  - Name all columns in SELECT — never use SELECT *

When results come back:
  - Explain what the numbers mean in plain English
  - Point out anything surprising or worth investigating
  - Suggest follow-up queries if relevant"""


# ── Step 1: One-time setup ────────────────────────────────────────────────────

def index_everything(
    trino: TrinoConnector,
    store: VectorStore,
    rel_registry: RelationshipRegistry,
):
    """
    Run ONCE to index all schemas + discover relationships.
    Re-run when tables are added/changed.
    """
    print("\n[1/2] Indexing schemas from Trino...")
    indexer = SchemaIndexer(trino, store)
    total = indexer.index_all(
        catalog=TRINO_CATALOG,
        skip_schemas=["information_schema", "sys"],
        include_samples=False,  # True = richer context, slower
        verbose=True,
    )
    print(f"     Indexed {total} chunks")

    print("\n[2/2] Discovering table relationships...")
    n = rel_registry.discover_from_store(store)
    print(f"     Found {n} inferred relationships")

    # Add explicit relationships you know about
    # rel_registry.add("hive.analytics.orders", "customer_id",
    #                  "hive.analytics.customers", "id")
    # rel_registry.add("hive.analytics.order_items", "product_id",
    #                  "hive.analytics.products", "id")

    print("\nSetup complete. You can now run the agent.")


# ── Step 2: Build agent ───────────────────────────────────────────────────────

def build_agent(
    trino: TrinoConnector,
    store: VectorStore,
    rel_registry: RelationshipRegistry,
    query_history: QueryHistory,
    use_reflection: bool = False,
) -> ReActAgent:

    cfg = Config()
    llm = cfg.build_llm()

    tools = ToolRegistry()
    register_builtin_tools(tools)
    register_nl_to_sql_tools(tools, trino, store)
    register_relationship_tools(tools, rel_registry)
    register_query_history_tools(tools, query_history)

    agent = ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=SQL_SYSTEM_PROMPT,
        max_iterations=15,
    )

    return agent, llm


# ── Step 3: Run ───────────────────────────────────────────────────────────────

def run_repl(agent: ReActAgent, reflector=None):
    print(f"\nNLP-to-SQL ready | {TRINO_HOST} | {TRINO_CATALOG}.{TRINO_SCHEMA}")
    print("Ask any question in plain English. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "reset":
            agent.reset()
            print("[Context cleared]")
            continue

        if reflector:
            initial = agent.run(question)
            result = reflector.reflect_and_refine(agent, question, initial, verbose=False)
            print(f"\nFinal answer:\n{result.final_answer}")
        else:
            agent.run(question)

        agent.reset()  # fresh context per question (avoids cross-contamination)


# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    # Schema discovery
    "What tables contain order and revenue data?",

    # Simple aggregate
    "How many orders were placed last month?",

    # Multi-table join
    "Show top 10 customers by total revenue in 2024",

    # Time-series analysis
    "Which product categories had declining revenue for 3+ consecutive months in 2024?",

    # Cohort analysis
    "What is the 30/60/90 day retention rate for users who signed up in Jan 2024?",

    # Complex CTE query
    """
    Find customers who:
    - Made their first purchase in Q1 2024
    - Then bought at least 3 more times
    - And whose average order value is above the overall average
    Show their names, total spend, and order count.
    """,

    # Just schema lookup (no execution)
    "Show me the schema of the sessions table",

    # Join path discovery
    "How do I join the events table to the users table?",
]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup",   action="store_true", help="Run one-time schema indexing")
    parser.add_argument("--reflect", action="store_true", help="Enable reflection for verified answers")
    parser.add_argument("--query",   type=str,            help="Run a single query and exit")
    args = parser.parse_args()

    # Connect to Trino
    trino = TrinoConnector(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=TRINO_USER,
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
    )

    # Shared stores
    store         = VectorStore(persist_dir=f"{MEMORY_DIR}/schema_rag", chunk_size=800)
    rel_registry  = RelationshipRegistry(path=f"{MEMORY_DIR}/relationships.json")
    query_history = QueryHistory(path=f"{MEMORY_DIR}/query_history.jsonl")

    if args.setup:
        index_everything(trino, store, rel_registry)
        sys.exit(0)

    # Build agent
    agent, llm = build_agent(trino, store, rel_registry, query_history)
    reflector = Reflector(llm, max_retries=1) if args.reflect else None

    if args.query:
        agent.run(args.query)
    else:
        run_repl(agent, reflector)
