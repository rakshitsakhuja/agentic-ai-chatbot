"""
Dynamic Tool Synthesis
──────────────────────
Lets the agent create new Python tools on the fly when no existing tool fits.

Flow:
  1. Agent calls create_tool(name, description, code, input_schema)
  2. Code is exec'd, function is registered in the live ToolRegistry
  3. Tool definition saved to .agent_tools/tools.json
  4. On next startup, load_saved_tools() restores all saved tools automatically

Also registers:
  - list_dynamic_tools()  → shows all saved tools
  - delete_dynamic_tool() → removes a tool by name

Security note:
  exec() runs in the same Python process — only use with trusted LLMs.
  For production, wrap in a subprocess sandbox or use RestrictedPython.
"""

import inspect
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict

from tools.registry import ToolRegistry

_TOOLS_FILE = os.path.join(".agent_tools", "tools.json")


# ── Persistence ───────────────────────────────────────────────────────────────

class DynamicToolStore:
    """
    Persists dynamically created tool definitions to disk as JSON.
    Survives process restarts — tools auto-reload on next session.
    """
    def __init__(self, tools_file: str = _TOOLS_FILE):
        self._file = tools_file
        os.makedirs(os.path.dirname(tools_file), exist_ok=True)
        self._tools: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if os.path.exists(self._file):
            try:
                with open(self._file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_disk(self):
        with open(self._file, "w") as f:
            json.dump(self._tools, f, indent=2)

    def save(self, name: str, description: str, code: str, input_schema: dict):
        self._tools[name] = {
            "name":         name,
            "description":  description,
            "code":         code,
            "input_schema": input_schema,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        }
        self._save_disk()

    def delete(self, name: str) -> bool:
        existed = name in self._tools
        self._tools.pop(name, None)
        self._save_disk()
        return existed

    def all(self) -> Dict[str, dict]:
        return dict(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ── Exec helper ───────────────────────────────────────────────────────────────

def _exec_fn(code: str, name: str):
    """Compile + exec tool code; return the function object."""
    namespace: dict = {}
    exec(compile(code, f"<dynamic:{name}>", "exec"), namespace)  # noqa: S102
    fn = namespace.get(name)
    if fn is None:
        raise ValueError(f"Code must define a function named '{name}'.")
    if not callable(fn):
        raise ValueError(f"'{name}' in the code is not callable.")
    return fn


def _infer_schema(fn) -> dict:
    """
    Auto-generate a basic JSON schema from a function's type annotations.
    Used as a fallback when the LLM doesn't provide one.
    """
    _PY_TO_JSON = {int: "integer", float: "number", bool: "boolean", str: "string"}
    sig = inspect.signature(fn)
    props, required = {}, []
    for param_name, param in sig.parameters.items():
        ann = param.annotation
        json_type = _PY_TO_JSON.get(ann, "string")
        props[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return {"type": "object", "properties": props, "required": required}


# ── Load saved tools on startup ───────────────────────────────────────────────

def load_saved_tools(registry: ToolRegistry, store: DynamicToolStore) -> int:
    """
    Restore all previously created tools into the live registry.
    Call this once at agent startup — before the first run().
    """
    loaded = 0
    for name, td in store.all().items():
        try:
            fn = _exec_fn(td["code"], name)
            registry.register(
                name=name,
                description=td["description"],
                input_schema=td["input_schema"],
                fn=fn,
                tags=["dynamic", "custom"],
            )
            loaded += 1
        except Exception as e:
            print(f"[DynamicTools] Could not load '{name}': {e}")
    if loaded:
        print(f"[DynamicTools] Restored {loaded} saved tool(s)")
    return loaded


# ── Register meta-tools ───────────────────────────────────────────────────────

def register_dynamic_tools(registry: ToolRegistry, store: DynamicToolStore):
    """
    Register the create_tool / list_dynamic_tools / delete_dynamic_tool
    meta-tools onto the given registry.
    """

    @registry.tool(
        description=(
            "Create a new Python tool and register it immediately. "
            "Saved to disk — auto-loads in future sessions. "
            "code must define a function named exactly as 'name', returning a string. "
            "input_schema is optional — inferred from type annotations if omitted. "
            "IMPORTANT: call list_dynamic_tools first — reuse an existing tool if it covers the same job. "
            "If overwriting an existing tool with the same name, it will be updated in place."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "snake_case function name describing WHAT the tool does generically, "
                        "NOT where it is used. "
                        "GOOD: 'scrape_url', 'fetch_html_text', 'parse_json_from_url'. "
                        "BAD: 'crawl_langchain_docs', 'get_github_readme' — these are task-specific, not reusable."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "One-sentence description shown to the agent on future runs",
                },
                "code": {
                    "type": "string",
                    "description": "Complete Python function definition",
                },
                "input_schema": {
                    "type": "object",
                    "description": (
                        "JSON schema for function parameters. "
                        "If omitted or {}, it will be inferred from type annotations."
                    ),
                },
            },
            "required": ["name", "description", "code"],
        },
        tags=["meta", "dynamic"],
    )
    def create_tool(
        name: str,
        description: str,
        code: str,
        input_schema: dict = None,
    ) -> str:
        # ── Validate name ──────────────────────────────────────────────────────
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            return (
                f"Invalid tool name '{name}'. "
                "Use lowercase snake_case (letters, digits, underscores, must start with a letter)."
            )
        if name in ("create_tool", "list_dynamic_tools", "delete_dynamic_tool"):
            return f"Cannot overwrite built-in meta-tool '{name}'."

        already_exists = name in store
        if name in registry.list_tools() and not already_exists:
            return (
                f"A built-in tool named '{name}' already exists. "
                "Choose a different name."
            )

        # ── Compile & exec ─────────────────────────────────────────────────────
        try:
            fn = _exec_fn(code, name)
        except SyntaxError as e:
            return f"Syntax error in code:\n{e}"
        except Exception as e:
            return f"Error defining tool:\n{e}"

        # ── Infer schema if missing ────────────────────────────────────────────
        if not input_schema:
            try:
                input_schema = _infer_schema(fn)
            except Exception:
                input_schema = {"type": "object", "properties": {}, "required": []}

        # ── Register live ──────────────────────────────────────────────────────
        registry.register(
            name=name,
            description=description,
            input_schema=input_schema,
            fn=fn,
            tags=["dynamic", "custom"],
        )

        # ── Persist ────────────────────────────────────────────────────────────
        store.save(name, description, code, input_schema)

        action = "updated" if already_exists else "created"
        return (
            f"Tool '{name}' {action} and registered successfully.\n"
            f"It is available right now in this session and will auto-load in all future sessions.\n"
            f"Use it like any other tool."
        )

    @registry.tool(
        description="List all dynamically created tools saved to disk.",
        input_schema={"type": "object", "properties": {}, "required": []},
        tags=["meta", "dynamic"],
    )
    def list_dynamic_tools() -> str:
        tools = store.all()
        if not tools:
            return "No dynamic tools created yet."
        lines = []
        for name, t in tools.items():
            lines.append(
                f"  {name}  —  {t['description'][:80]}"
                f"  (created {t['created_at'][:10]})"
            )
        return f"{len(tools)} dynamic tool(s):\n" + "\n".join(lines)

    @registry.tool(
        description="Delete a dynamically created tool by name. Removes it from disk permanently.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the tool to delete"},
            },
            "required": ["name"],
        },
        tags=["meta", "dynamic"],
    )
    def delete_dynamic_tool(name: str) -> str:
        if store.delete(name):
            registry.unregister(name)
            return f"Tool '{name}' deleted from disk and unregistered."
        return f"No dynamic tool named '{name}' found."
