"""
Built-in Tools
──────────────
Generic tools every agent can use out of the box.
Register them with your ToolRegistry instance:

    from tools.builtin import register_builtin_tools
    register_builtin_tools(registry)

For domain-specific tools (trading, data, etc.) create a separate file
e.g. tools/trading.py and register them the same way.
"""

from .registry import ToolRegistry


def register_builtin_tools(registry: ToolRegistry):
    """Register all built-in tools onto the given registry."""

    # ── Shell ─────────────────────────────────────────────────────────────────

    @registry.tool(
        description="Run a shell command. Returns stdout, stderr, and exit code.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout seconds (default 30)", "default": 30},
            },
            "required": ["command"],
        },
        tags=["system"],
    )
    def run_shell(command: str, timeout: int = 30) -> str:
        import subprocess
        try:
            r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
            parts = []
            if r.stdout.strip():
                parts.append(f"STDOUT:\n{r.stdout.strip()}")
            if r.stderr.strip():
                parts.append(f"STDERR:\n{r.stderr.strip()}")
            parts.append(f"exit_code={r.returncode}")
            return "\n".join(parts) or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"

    # ── File I/O ──────────────────────────────────────────────────────────────

    @registry.tool(
        description="Read a file and return its contents (up to max_lines).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_lines": {"type": "integer", "default": 300},
            },
            "required": ["path"],
        },
        tags=["system"],
    )
    def read_file(path: str, max_lines: int = 300) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            trimmed = lines[:max_lines]
            text = "".join(trimmed)
            if len(lines) > max_lines:
                text += f"\n[...{len(lines) - max_lines} more lines truncated]"
            return text
        except FileNotFoundError:
            return f"ERROR: not found: {path}"
        except Exception as e:
            return f"ERROR: {e}"

    @registry.tool(
        description="Write content to a file. Creates parent directories if needed.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        },
        tags=["system"],
    )
    def write_file(path: str, content: str, append: bool = False) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content)} chars to {path}"

    # ── Search ────────────────────────────────────────────────────────────────

    @registry.tool(
        description="Search for a regex pattern recursively across files.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex to search"},
                "directory": {"type": "string", "default": "."},
                "file_glob": {"type": "string", "default": "**/*"},
                "max_results": {"type": "integer", "default": 40},
            },
            "required": ["pattern"],
        },
        tags=["system"],
    )
    def search_files(pattern: str, directory: str = ".", file_glob: str = "**/*", max_results: int = 40) -> str:
        import re, glob, os
        hits = []
        for filepath in glob.glob(os.path.join(directory, file_glob), recursive=True):
            if not os.path.isfile(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            hits.append(f"{filepath}:{i}: {line.rstrip()}")
                            if len(hits) >= max_results:
                                return "\n".join(hits) + f"\n[truncated at {max_results}]"
            except Exception:
                pass
        return "\n".join(hits) if hits else "No matches"

    # ── HTTP ──────────────────────────────────────────────────────────────────

    @registry.tool(
        description="Make an HTTP GET or POST request and return the response body.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET", "enum": ["GET", "POST", "PUT", "DELETE"]},
                "headers": {"type": "object", "default": {}},
                "body": {"type": "object", "description": "JSON body for POST/PUT", "default": {}},
                "timeout": {"type": "integer", "default": 15},
            },
            "required": ["url"],
        },
        tags=["network"],
    )
    def http_request(url: str, method: str = "GET", headers: dict = {}, body: dict = {}, timeout: int = 15) -> str:
        import urllib.request, urllib.error, json as _json
        req = urllib.request.Request(url, method=method)
        for k, v in headers.items():
            req.add_header(k, v)
        data = None
        if body and method in ("POST", "PUT"):
            data = _json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
                raw = resp.read().decode(errors="replace")
                # Try to pretty-print JSON
                try:
                    return _json.dumps(_json.loads(raw), indent=2)[:4000]
                except Exception:
                    return raw[:4000]
        except urllib.error.HTTPError as e:
            return f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            return f"ERROR: {e}"

    # ── Python REPL ───────────────────────────────────────────────────────────

    @registry.tool(
        description=(
            "Execute Python code in an isolated namespace and return printed output + return value. "
            "Useful for quick calculations, data transforms, analysis."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run"},
            },
            "required": ["code"],
        },
        tags=["system", "compute"],
    )
    def python_repl(code: str) -> str:
        import io, contextlib
        stdout = io.StringIO()
        ns: dict = {}
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<agent_repl>", "exec"), ns)
            output = stdout.getvalue()
            # If last expression has a value, show it
            last_line = code.strip().splitlines()[-1] if code.strip() else ""
            result = ""
            if not last_line.startswith(("#", "import", "from", "def ", "class ", "if ", "for ", "while ", "try")):
                try:
                    val = eval(last_line, ns)
                    if val is not None:
                        result = f"\n=> {repr(val)}"
                except Exception:
                    pass
            return (output + result).strip() or "(no output)"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
