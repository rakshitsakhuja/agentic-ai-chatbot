"""
Table Relationship Discovery
──────────────────────────────
With 500+ tables the agent needs to know HOW to join them.
This module infers foreign-key relationships from column naming patterns
and stores them so the agent can ask "how do I join orders to customers?".

Discovery methods:
  1. Column name pattern matching  — user_id in orders → users.id
  2. Explicit relationship registry — you define joins manually
  3. Information schema (if available) — reads FK constraints from Trino

Usage:
    rel = RelationshipRegistry(".agent_memory/relationships.json")

    # Auto-discover from indexed schemas
    rel.discover_from_store(store)

    # Or define manually
    rel.add("hive.analytics.orders", "customer_id", "hive.analytics.customers", "id")
    rel.add("hive.analytics.order_items", "order_id", "hive.analytics.orders", "id")

    # Register as agent tool
    register_relationship_tools(registry, rel)

    # Agent calls:
    #   find_join_path("orders", "products")
    #     → "orders → order_items (order_id=order_id) → products (product_id=id)"
"""

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple, Set
from collections import defaultdict, deque


@dataclass
class Relationship:
    from_table: str      # fully qualified: catalog.schema.table
    from_col: str
    to_table: str        # fully qualified
    to_col: str
    confidence: str      # "explicit" | "inferred"


class RelationshipRegistry:
    def __init__(self, path: str = ".agent_memory/relationships.json"):
        self._path = path
        self._rels: List[Relationship] = []
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    # ── Building the graph ────────────────────────────────────────────────────

    def add(
        self,
        from_table: str, from_col: str,
        to_table: str, to_col: str,
        confidence: str = "explicit",
    ):
        """Manually define a relationship."""
        rel = Relationship(from_table, from_col, to_table, to_col, confidence)
        # avoid duplicates
        for r in self._rels:
            if (r.from_table == from_table and r.from_col == from_col
                    and r.to_table == to_table):
                return
        self._rels.append(rel)
        self._save()

    def discover_from_store(self, store, min_confidence: float = 0.0):
        """
        Auto-infer relationships from indexed schemas using column name patterns.

        Rules:
          - col named `<table>_id` in table A → likely FK to table B where B matches <table>
          - col named `id` is likely a PK
          - col named `<something>_id` → search for a table named <something>
        """
        from tools.rag import VectorStore

        # Build index: table_name (last part) → full qualified name
        table_index: Dict[str, List[str]] = defaultdict(list)
        for chunk in store._chunks:
            meta = chunk.metadata
            if "table" in meta:
                table_index[meta["table"].lower()].append(chunk.doc_name)

        # Parse column names from chunk text
        for chunk in store._chunks:
            if "Columns:" not in chunk.text:
                continue
            from_table = chunk.doc_name  # catalog.schema.table
            col_section = chunk.text.split("Columns:")[-1]
            for line in col_section.splitlines():
                line = line.strip()
                if not line or line.startswith("Column names:"):
                    continue
                # Extract column name: "  col_name (type)"
                m = re.match(r'(\w+)\s+\(', line)
                if not m:
                    continue
                col_name = m.group(1).lower()

                # Pattern: ends with _id (but not just "id")
                if col_name.endswith("_id") and col_name != "id":
                    ref_table = col_name[:-3]  # strip _id
                    if ref_table in table_index:
                        for to_table in table_index[ref_table]:
                            if to_table != from_table:
                                self.add(
                                    from_table, col_name,
                                    to_table, "id",
                                    confidence="inferred",
                                )

        self._save()
        return len(self._rels)

    # ── Querying ──────────────────────────────────────────────────────────────

    def find_direct_joins(self, table: str) -> List[Relationship]:
        """Return all relationships involving a table (from or to)."""
        t = table.lower()
        return [
            r for r in self._rels
            if t in r.from_table.lower() or t in r.to_table.lower()
        ]

    def find_join_path(self, from_table: str, to_table: str, max_hops: int = 4) -> Optional[List[Relationship]]:
        """
        BFS to find shortest join path between two tables.
        Returns list of relationships forming the path, or None if no path found.
        """
        # Build adjacency
        adj: Dict[str, List[Tuple[str, Relationship]]] = defaultdict(list)
        for r in self._rels:
            adj[r.from_table].append((r.to_table, r))
            adj[r.to_table].append((r.from_table, r))  # bidirectional

        # Normalise table names for matching
        def match(a: str, b: str) -> bool:
            return b.lower() in a.lower() or a.lower() in b.lower()

        # Find start nodes
        start_nodes = [t for t in adj if match(t, from_table)]
        end_nodes = [t for t in adj if match(t, to_table)]

        if not start_nodes or not end_nodes:
            return None

        # BFS
        queue = deque([(start_nodes[0], [])])
        visited: Set[str] = {start_nodes[0]}

        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for neighbor, rel in adj[node]:
                if any(match(neighbor, e) for e in end_nodes):
                    return path + [rel]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [rel]))
        return None

    def format_path(self, path: List[Relationship]) -> str:
        if not path:
            return "No join path found."
        parts = []
        for r in path:
            parts.append(
                f"{r.from_table} JOIN {r.to_table} ON "
                f"{r.from_table}.{r.from_col} = {r.to_table}.{r.to_col}"
            )
        return "\n".join(parts)

    def format_direct_joins(self, rels: List[Relationship]) -> str:
        if not rels:
            return "No known relationships for this table."
        lines = []
        for r in rels:
            lines.append(
                f"  {r.from_table}.{r.from_col} → {r.to_table}.{r.to_col} [{r.confidence}]"
            )
        return "\n".join(lines)

    def _save(self):
        with open(self._path, "w") as f:
            json.dump([asdict(r) for r in self._rels], f, indent=2)

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                    self._rels = [Relationship(**d) for d in data]
            except Exception:
                pass


def register_relationship_tools(registry, rel_registry: RelationshipRegistry):

    @registry.tool(
        description=(
            "Find how to JOIN two tables together. "
            "Returns the join path with ON conditions. "
            "Call this when you need to join tables but aren't sure which columns to use."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "from_table": {"type": "string", "description": "Starting table name"},
                "to_table":   {"type": "string", "description": "Target table to join to"},
            },
            "required": ["from_table", "to_table"],
        },
        tags=["sql", "schema"],
    )
    def find_join_path(from_table: str, to_table: str) -> str:
        path = rel_registry.find_join_path(from_table, to_table)
        if path:
            return f"Join path found:\n{rel_registry.format_path(path)}"
        return (
            f"No known join path between '{from_table}' and '{to_table}'.\n"
            "Try searching schemas for shared column names manually."
        )

    @registry.tool(
        description=(
            "Show all known JOIN relationships for a table — "
            "which other tables it connects to and on which columns."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name to look up"},
            },
            "required": ["table"],
        },
        tags=["sql", "schema"],
    )
    def get_table_relationships(table: str) -> str:
        rels = rel_registry.find_direct_joins(table)
        return rel_registry.format_direct_joins(rels)
