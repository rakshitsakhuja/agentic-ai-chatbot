"""
Short-Term Memory (Conversation Window)
────────────────────────────────────────
Sliding-window conversation history.
Keeps the last N messages so context never exceeds the model's limit.

For agentic loops the conversation grows with tool calls/results —
a sliding window prevents token overflow while preserving recent context.
"""

from typing import List, Optional
from llm.base import Message


class ShortTermMemory:
    def __init__(self, max_messages: int = 40):
        """
        max_messages: max non-system messages to keep.
        System message is always kept.
        """
        self.max_messages = max_messages
        self._system: Optional[Message] = None
        self._history: List[Message] = []

    def set_system(self, content: str):
        self._system = Message(role="system", content=content)

    def add(self, message: Message):
        self._history.append(message)
        self._trim()

    def add_user(self, content: str):
        self.add(Message(role="user", content=content))

    def add_assistant(self, content, tool_calls=None):
        """
        content: str text or list of content blocks (for Anthropic tool_use blocks)
        tool_calls: list of ToolCall objects when the assistant made tool calls
        """
        self.add(Message(role="assistant", content=content, tool_calls=tool_calls or []))

    def add_tool_result(self, tool_call_id: str, content: str, tool_name: str = ""):
        self.add(Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
        ))

    def get_messages(self) -> List[Message]:
        """Return full message list (system + history)."""
        if self._system:
            return [self._system] + self._history
        return list(self._history)

    def _trim(self):
        if len(self._history) <= self.max_messages:
            return

        # Remove at least `excess` messages from the front, but advance the
        # cut point to the next 'user' message so we never slice through an
        # assistant+tool_results group.  Splitting such a group leaves orphaned
        # tool messages (role='tool' with no preceding tool_calls) which causes
        # 400 errors on every subsequent API call.
        excess = len(self._history) - self.max_messages
        cut = excess
        while cut < len(self._history) and self._history[cut].role != "user":
            cut += 1
        # If no user message found beyond the cut, hard-trim as a last resort.
        self._history = (
            self._history[cut:]
            if cut < len(self._history)
            else self._history[-self.max_messages :]
        )

    def clear(self):
        self._history = []

    def token_estimate(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        total = sum(
            len(str(m.content)) for m in self.get_messages()
        )
        return total // 4

    def __len__(self) -> int:
        return len(self._history)
