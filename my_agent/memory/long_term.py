"""
Long-Term Memory
────────────────
Two-tier persistent memory:

1. KVStore  — fast key→value lookup (JSON file on disk)
              Good for: user preferences, known facts, last-seen prices,
                        entity properties, configuration overrides.

2. EpisodicStore — timestamped episode log with keyword search.
                   Good for: past decisions, observations, outcomes,
                              event history, trade logs.

No external vector DB needed — uses simple keyword scoring for search.
Drop in ChromaDB / Pinecone / pgvector for production-grade semantic search.
"""

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# ── KV Store ──────────────────────────────────────────────────────────────────

class KVStore:
    def __init__(self, path: str = ".agent_memory/kv.json"):
        self._path = path
        self._data: Dict[str, Any] = {}
        self._load()

    def store(self, key: str, value: Any):
        self._data[key] = value
        self._save()

    def recall(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str):
        self._data.pop(key, None)
        self._save()

    def list_keys(self) -> List[str]:
        return list(self._data.keys())

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)


# ── Episodic Store ────────────────────────────────────────────────────────────

@dataclass
class Episode:
    id: str
    timestamp: float
    category: str        # e.g. "trade", "observation", "decision", "error"
    content: str
    metadata: Dict[str, Any]


class EpisodicStore:
    def __init__(self, path: str = ".agent_memory/episodes.jsonl"):
        self._path = path
        self._episodes: List[Episode] = []
        self._load()

    def record(
        self,
        content: str,
        category: str = "general",
        metadata: Optional[Dict] = None,
    ) -> Episode:
        ep = Episode(
            id=f"ep_{int(time.time() * 1000)}",
            timestamp=time.time(),
            category=category,
            content=content,
            metadata=metadata or {},
        )
        self._episodes.append(ep)
        self._append(ep)
        return ep

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Episode]:
        """Keyword-based search. Replace with embeddings for production."""
        keywords = set(query.lower().split())
        scored = []
        for ep in self._episodes:
            if category and ep.category != category:
                continue
            text = ep.content.lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [ep for _, ep in scored[:top_k]]

    def recent(self, n: int = 10, category: Optional[str] = None) -> List[Episode]:
        filtered = [
            ep for ep in self._episodes
            if not category or ep.category == category
        ]
        return filtered[-n:]

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            d = json.loads(line)
                            self._episodes.append(Episode(**d))
            except Exception:
                pass

    def _append(self, ep: Episode):
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(ep), default=str) + "\n")
