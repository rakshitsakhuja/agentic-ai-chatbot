"""
NLP-to-SQL Tools for Trino (500+ tables)
─────────────────────────────────────────
Core insight: with 500+ tables you CANNOT put all schemas in context.
Solution: Schema RAG — index all schemas once, retrieve relevant ones per query.

Pipeline:
  1. SchemaIndexer  — one-time job: extracts all schemas from Trino → VectorStore
  2. search_schema  — agent tool: finds relevant tables for a query
  3. get_table_schema — agent tool: gets full column details for a table
  4. execute_sql    — agent tool: runs SQL on Trino, returns results
  5. explain_sql    — agent tool: validates SQL before running (saves quota)
  6. get_sample_data — agent tool: shows 5 rows so agent understands data shape

Usage:
    from tools.nl_to_sql import TrinoConnector, SchemaIndexer, register_nl_to_sql_tools

    trino = TrinoConnector(host="trino.mycompany.com", port=443, user="analyst")
    store = VectorStore(persist_dir=".agent_memory/schema_rag")

    # One-time: index all schemas (run again when schemas change)
    indexer = SchemaIndexer(trino, store)
    indexer.index_all(catalog="hive")   # or index specific schemas

    # Register tools
    register_nl_to_sql_tools(registry, trino, store)

    # Agent can now answer:
    # "How many orders were placed last month per region?"
    # "Show me the top 10 customers by revenue in Q4 2024"
    # "Which products have declining sales for 3+ consecutive months?"
"""

import os
import json
import time
from typing import List, Optional, Dict, Any

from tools.registry import ToolRegistry
from tools.rag import VectorStore


# ── Trino Connector ───────────────────────────────────────────────────────────

class TrinoConnector:
    """
    Wraps trino Python client.
    Install: pip install trino
    """

    def __init__(
        self,
        host: str,
        port: int = 443,
        user: str = "agent",
        catalog: str = "hive",
        schema: str = "default",
        http_scheme: str = "https",
        auth=None,               # trino.auth.BasicAuthentication("user", "pass")
        extra_headers: dict = None,
    ):
        try:
            import trino
        except ImportError:
            raise ImportError("Run: pip install trino")

        self.default_catalog = catalog
        self.default_schema = schema

        self._conn_kwargs = dict(
            host=host,
            port=port,
            user=user,
            catalog=catalog,
            schema=schema,
            http_scheme=http_scheme,
        )
        if auth:
            self._conn_kwargs["auth"] = auth
        if extra_headers:
            self._conn_kwargs["http_headers"] = extra_headers

    def _connect(self):
        import trino
        return trino.dbapi.connect(**self._conn_kwargs)

    def execute(
        self,
        sql: str,
        limit: int = 200,
        timeout_s: int = 60,
    ) -> Dict[str, Any]:
        """
        Run a SQL query. Returns:
        {
            "columns": ["col1", "col2"],
            "rows": [[...], [...]],
            "row_count": 42,
            "truncated": False,
        }
        """
        conn = self._connect()
        cur = conn.cursor()
        try:
            start = time.time()
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(limit)
            elapsed = round(time.time() - start, 2)
            return {
                "columns": columns,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
                "truncated": len(rows) == limit,
                "elapsed_s": elapsed,
            }
        finally:
            cur.close()
            conn.close()

    def get_catalogs(self) -> List[str]:
        result = self.execute("SHOW CATALOGS")
        return [r[0] for r in result["rows"]]

    def get_schemas(self, catalog: str = None) -> List[str]:
        cat = catalog or self.default_catalog
        result = self.execute(f"SHOW SCHEMAS FROM {cat}")
        return [r[0] for r in result["rows"]]

    def get_tables(self, schema: str, catalog: str = None) -> List[str]:
        cat = catalog or self.default_catalog
        result = self.execute(f"SHOW TABLES FROM {cat}.{schema}")
        return [r[0] for r in result["rows"]]

    def get_columns(self, table: str, schema: str = None, catalog: str = None) -> List[Dict]:
        """Returns list of {name, type, comment} for each column."""
        cat = catalog or self.default_catalog
        sch = schema or self.default_schema
        result = self.execute(f"DESCRIBE {cat}.{sch}.{table}")
        cols = []
        for row in result["rows"]:
            cols.append({
                "name":    row[0],
                "type":    row[1],
                "extra":   row[2] if len(row) > 2 else "",
                "comment": row[3] if len(row) > 3 else "",
            })
        return cols

    def get_sample(self, table: str, schema: str = None, catalog: str = None, n: int = 3) -> Dict:
        cat = catalog or self.default_catalog
        sch = schema or self.default_schema
        return self.execute(f"SELECT * FROM {cat}.{sch}.{table} LIMIT {n}")

    def table_stats(self, table: str, schema: str = None, catalog: str = None) -> str:
        """Try to get row count (may not work on all Trino connectors)."""
        cat = catalog or self.default_catalog
        sch = schema or self.default_schema
        try:
            result = self.execute(f"SELECT COUNT(*) FROM {cat}.{sch}.{table}")
            return str(result["rows"][0][0]) if result["rows"] else "unknown"
        except Exception:
            return "unknown"

    def format_result(self, result: Dict, max_rows: int = 50) -> str:
        """Format query result as a readable table string."""
        if not result["columns"]:
            return "(no results)"

        cols = result["columns"]
        rows = result["rows"][:max_rows]

        # Column widths
        widths = [len(c) for c in cols]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header = "|" + "|".join(f" {c:<{widths[i]}} " for i, c in enumerate(cols)) + "|"

        lines = [sep, header, sep]
        for row in rows:
            line = "|" + "|".join(f" {str(v):<{widths[i]}} " for i, v in enumerate(row)) + "|"
            lines.append(line)
        lines.append(sep)

        summary = f"{result['row_count']} rows"
        if result.get("truncated"):
            summary += f" (truncated at {max_rows})"
        summary += f" | {result.get('elapsed_s', '?')}s"

        return "\n".join(lines) + f"\n{summary}"


# ── Schema Indexer ────────────────────────────────────────────────────────────

class SchemaIndexer:
    """
    One-time job: crawls Trino, extracts all table schemas,
    stores them in VectorStore for semantic search.

    Run this:
    - Once on setup
    - When new tables are added
    - On a schedule (weekly/daily) to stay fresh
    """

    def __init__(self, trino: TrinoConnector, store: VectorStore):
        self.trino = trino
        self.store = store

    def index_all(
        self,
        catalog: str = None,
        schemas: List[str] = None,       # None = all schemas
        skip_schemas: List[str] = None,  # e.g. ["information_schema"]
        include_samples: bool = False,   # adds 3 sample rows per table (richer but slower)
        verbose: bool = True,
    ) -> int:
        """
        Index all tables. Returns total chunks stored.
        With 500 tables expect ~2-10 min depending on Trino speed.
        """
        cat = catalog or self.trino.default_catalog
        skip = set(skip_schemas or ["information_schema", "sys"])
        schema_list = schemas or self.trino.get_schemas(cat)
        schema_list = [s for s in schema_list if s not in skip]

        total = 0
        for schema in schema_list:
            if verbose:
                print(f"  Indexing schema: {schema}")
            try:
                tables = self.trino.get_tables(schema, cat)
            except Exception as e:
                print(f"    Skip {schema}: {e}")
                continue

            for table in tables:
                try:
                    n = self.index_table(cat, schema, table, include_samples)
                    total += n
                    if verbose:
                        print(f"    + {schema}.{table} ({n} chunks)")
                except Exception as e:
                    print(f"    ! {schema}.{table} failed: {e}")

        if verbose:
            print(f"\nIndexed {total} chunks across {len(schema_list)} schemas")
        return total

    def index_table(
        self,
        catalog: str,
        schema: str,
        table: str,
        include_samples: bool = False,
    ) -> int:
        """Index a single table. Returns chunks stored."""
        cols = self.trino.get_columns(table, schema, catalog)

        # Build rich text description for semantic search
        lines = [
            f"Table: {catalog}.{schema}.{table}",
            f"Full name: {catalog}.{schema}.{table}",
            f"Schema: {schema}",
            f"Catalog: {catalog}",
            "",
            "Columns:",
        ]
        for col in cols:
            comment = f" -- {col['comment']}" if col.get("comment") else ""
            lines.append(f"  {col['name']} ({col['type']}){comment}")

        # Add searchable keywords: column names are very important for retrieval
        col_names = ", ".join(c["name"] for c in cols)
        lines.append(f"\nColumn names: {col_names}")

        if include_samples:
            try:
                sample = self.trino.get_sample(table, schema, catalog, n=3)
                if sample["rows"]:
                    lines.append("\nSample data (3 rows):")
                    lines.append("  Columns: " + ", ".join(sample["columns"]))
                    for row in sample["rows"]:
                        lines.append("  " + str(row))
            except Exception:
                pass

        text = "\n".join(lines)
        return self.store.ingest_text(
            text=text,
            doc_name=f"{catalog}.{schema}.{table}",
            metadata={"catalog": catalog, "schema": schema, "table": table},
        )

    def index_single(self, full_table_name: str, include_samples: bool = False) -> int:
        """Index a single table by full name: catalog.schema.table"""
        parts = full_table_name.split(".")
        if len(parts) == 3:
            cat, schema, table = parts
        elif len(parts) == 2:
            schema, table = parts
            cat = self.trino.default_catalog
        else:
            raise ValueError(f"Invalid table name: {full_table_name}. Use catalog.schema.table")
        return self.index_table(cat, schema, table, include_samples)


# ── Register agent tools ──────────────────────────────────────────────────────

def register_nl_to_sql_tools(
    registry: ToolRegistry,
    trino: TrinoConnector,
    store: VectorStore,
):
    """Register all NLP-to-SQL tools onto the agent's ToolRegistry."""

    @registry.tool(
        description=(
            "Search the schema index to find tables relevant to a question. "
            "ALWAYS call this first before writing any SQL. "
            "Returns table names + column descriptions for the most relevant tables."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of what data you need. "
                        "Be specific — mention entities, metrics, time ranges."
                    ),
                },
                "top_k": {"type": "integer", "default": 6, "description": "Number of tables to return"},
            },
            "required": ["query"],
        },
        tags=["sql", "schema"],
    )
    def search_schema(query: str, top_k: int = 6) -> str:
        results = store.search(query, top_k=top_k)
        if not results:
            return (
                "No tables found in schema index. "
                "Run SchemaIndexer.index_all() to index your Trino tables first."
            )
        parts = []
        for score, chunk in results:
            parts.append(f"[relevance={score:.2f}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    @registry.tool(
        description="Get the full column details for a specific table.",
        input_schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name (catalog.schema.table or schema.table)"},
            },
            "required": ["table"],
        },
        tags=["sql", "schema"],
    )
    def get_table_schema(table: str) -> str:
        parts = table.split(".")
        if len(parts) == 3:
            cat, schema, tbl = parts
        elif len(parts) == 2:
            schema, tbl = parts
            cat = trino.default_catalog
        else:
            schema = trino.default_schema
            tbl = parts[0]
            cat = trino.default_catalog
        try:
            cols = trino.get_columns(tbl, schema, cat)
            lines = [f"Table: {cat}.{schema}.{tbl}", "Columns:"]
            for c in cols:
                comment = f"  -- {c['comment']}" if c.get("comment") else ""
                lines.append(f"  {c['name']}  {c['type']}{comment}")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR fetching schema for {table}: {e}"

    @registry.tool(
        description="Get a few sample rows from a table to understand its data shape and values.",
        input_schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "catalog.schema.table"},
                "n": {"type": "integer", "default": 5, "description": "Number of rows"},
            },
            "required": ["table"],
        },
        tags=["sql", "schema"],
    )
    def get_sample_data(table: str, n: int = 5) -> str:
        parts = table.split(".")
        if len(parts) == 3:
            cat, schema, tbl = parts
        elif len(parts) == 2:
            schema, tbl = parts
            cat = trino.default_catalog
        else:
            schema = trino.default_schema
            tbl = parts[0]
            cat = trino.default_catalog
        try:
            result = trino.get_sample(tbl, schema, cat, n)
            return trino.format_result(result)
        except Exception as e:
            return f"ERROR getting sample from {table}: {e}"

    @registry.tool(
        description="List all schemas in a catalog.",
        input_schema={
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "description": "Catalog name (optional)"},
            },
            "required": [],
        },
        tags=["sql", "schema"],
    )
    def list_schemas(catalog: str = None) -> str:
        try:
            schemas = trino.get_schemas(catalog)
            return "\n".join(schemas)
        except Exception as e:
            return f"ERROR: {e}"

    @registry.tool(
        description="List all tables in a schema.",
        input_schema={
            "type": "object",
            "properties": {
                "schema": {"type": "string", "description": "Schema name"},
                "catalog": {"type": "string", "description": "Catalog name (optional)"},
            },
            "required": ["schema"],
        },
        tags=["sql", "schema"],
    )
    def list_tables(schema: str, catalog: str = None) -> str:
        try:
            tables = trino.get_tables(schema, catalog)
            return "\n".join(tables) if tables else "No tables found"
        except Exception as e:
            return f"ERROR: {e}"

    @registry.tool(
        description=(
            "Execute a SQL query on Trino and return the results. "
            "If the query fails, the error message will help you fix the SQL. "
            "Always use fully qualified table names: catalog.schema.table"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute. Use Trino SQL syntax.",
                },
                "limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "Max rows to return (default 100, max 500)",
                },
            },
            "required": ["sql"],
        },
        tags=["sql", "execute"],
    )
    def execute_sql(sql: str, limit: int = 100) -> str:
        limit = min(limit, 500)  # hard cap
        try:
            result = trino.execute(sql, limit=limit)
            formatted = trino.format_result(result)
            return formatted
        except Exception as e:
            return f"SQL ERROR: {e}\n\nFix the SQL and try again."

    @registry.tool(
        description=(
            "Validate and explain a SQL query WITHOUT running it. "
            "Use this to check syntax before executing expensive queries."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to explain"},
            },
            "required": ["sql"],
        },
        tags=["sql"],
    )
    def explain_sql(sql: str) -> str:
        try:
            result = trino.execute(f"EXPLAIN {sql}", limit=50)
            return "\n".join(str(r[0]) for r in result["rows"])
        except Exception as e:
            return f"EXPLAIN ERROR: {e}"
