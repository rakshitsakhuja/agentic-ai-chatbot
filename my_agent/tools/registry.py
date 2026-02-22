"""
Tool Registry
─────────────
Register Python functions as agent-callable tools.
Tools are auto-converted to the universal schema (Anthropic-style input_schema)
which all providers understand via their respective adapters.

Usage:
    registry = ToolRegistry()

    @registry.tool(
        description="Fetch current price of a token",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Token symbol e.g. BTC"},
            },
            "required": ["symbol"],
        }
    )
    def get_price(symbol: str) -> str:
        ...
"""

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict
    fn: Callable
    tags: List[str]  # e.g. ["trading", "data"] — for filtering tools per agent


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def tool(
        self,
        description: str,
        input_schema: Dict,
        tags: Optional[List[str]] = None,
    ):
        """Decorator — register a function as a callable tool."""
        def decorator(fn: Callable):
            self.register(
                name=fn.__name__,
                description=description,
                input_schema=input_schema,
                fn=fn,
                tags=tags or [],
            )
            return fn
        return decorator

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict,
        fn: Callable,
        tags: Optional[List[str]] = None,
    ):
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            fn=fn,
            tags=tags or [],
        )

    def unregister(self, name: str):
        self._tools.pop(name, None)

    # ── Querying ──────────────────────────────────────────────────────────────

    def list_tools(self, tags: Optional[List[str]] = None) -> List[str]:
        """List all tool names, optionally filtered by tags."""
        if not tags:
            return list(self._tools.keys())
        return [
            name for name, t in self._tools.items()
            if any(tag in t.tags for tag in tags)
        ]

    def as_schema(
        self,
        tags: Optional[List[str]] = None,
        names: Optional[set] = None,
    ) -> List[Dict]:
        """
        Return tools in universal schema format.
        Both Anthropic and OpenAI adapters accept this.

        Args:
            tags:  Only include tools whose tags overlap with this list.
            names: Explicit allowlist of tool names (takes priority over tags).
                   Pass the output of ToolRouter.select() here.
        """
        all_names = self.list_tools(tags)
        if names is not None:
            all_names = [n for n in all_names if n in names]
        return [
            {
                "name": self._tools[n].name,
                "description": self._tools[n].description,
                "input_schema": self._tools[n].input_schema,
            }
            for n in all_names
        ]

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, tool_call_id: str, inputs: Dict) -> ToolResult:
        """Execute a tool by name with given inputs. Always returns ToolResult."""
        if tool_name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call_id,
                content=f"Tool '{tool_name}' not found. Available: {self.list_tools()}",
                is_error=True,
            )
        try:
            raw = self._tools[tool_name].fn(**(inputs or {}))
            content = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            return ToolResult(tool_call_id=tool_call_id, content=content)
        except Exception:
            return ToolResult(
                tool_call_id=tool_call_id,
                content=traceback.format_exc(),
                is_error=True,
            )
