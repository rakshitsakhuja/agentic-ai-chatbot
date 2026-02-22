"""
RAG (Retrieval-Augmented Generation) Tools
───────────────────────────────────────────
Plugs directly into your existing ToolRegistry.

Two components:
  1. VectorStore  — stores document chunks + embeddings, handles search
  2. RAG tools    — ingest_document, search_knowledge_base, list_documents
                    registered onto your ToolRegistry so the agent can call them

Embedding backends (pick one):
  A. OpenAI embeddings  — best quality, needs openai package + API key
  B. sentence-transformers — local, free, good quality, needs torch
  C. TF-IDF fallback    — zero dependencies, always works, lower quality

Usage:
    from tools.rag import VectorStore, register_rag_tools

    store = VectorStore(persist_dir=".agent_memory/rag")
    register_rag_tools(registry, store)

    # Agent can now call:
    # ingest_document(path="report.pdf")
    # search_knowledge_base(query="what is the revenue?")
    # list_documents()
"""

import os
import re
import json
import math
import hashlib
import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from tools.registry import ToolRegistry


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)


# ── Embedding backends ────────────────────────────────────────────────────────

class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedder(BaseEmbedder):
    """Best quality. Needs: pip install openai"""
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in response.data]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local, free, good quality. Needs: pip install sentence-transformers"""
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()


class BM25Retriever:
    """
    BM25 ranking — better than TF-IDF as a zero-dependency fallback.
    Adds term frequency saturation (k1) and document length normalisation (b),
    which TF-IDF lacks. Used by Elasticsearch, Lucene, and most modern search engines.

    k1=1.5  controls how much repeated terms boost the score (saturates quickly)
    b=0.75  controls document length penalty (1.0 = full normalisation)

    Not an embedder — scores documents directly against a query at search time.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[List[str]] = []
        self._avgdl: float = 0.0
        self._idf: dict = {}

    def fit(self, texts: List[str]):
        self._corpus = [self._tok(t) for t in texts]
        self._avgdl = sum(len(d) for d in self._corpus) / max(len(self._corpus), 1)
        df: dict = {}
        for doc in self._corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        N = len(self._corpus)
        self._idf = {
            t: math.log((N - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def get_scores(self, query: str) -> List[float]:
        qtokens = self._tok(query)
        results = []
        for doc in self._corpus:
            dl = len(doc)
            tf_map: dict = {}
            for term in doc:
                tf_map[term] = tf_map.get(term, 0) + 1
            s = 0.0
            for term in qtokens:
                if term not in self._idf:
                    continue
                tf = tf_map.get(term, 0)
                s += self._idf[term] * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                )
            results.append(s)
        return results

    @staticmethod
    def _tok(text: str) -> List[str]:
        import re
        return re.findall(r'\b\w+\b', text.lower())


# ── Semantic Chunker ──────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Sentence-aware chunker with two backends — best available is used automatically:

    Backend 1 (preferred): semantic-text-splitter (Rust)
      pip install semantic-text-splitter
      - Recursive Unicode boundary splitting: chars → words → sentences → newlines
      - Handles abbreviations (e.g., et al.) correctly via Unicode spec, not regex
      - Rust speed — ~10x faster than Python regex on large papers
      - MarkdownSplitter used when text looks like markdown

    Backend 2 (fallback): pure Python regex
      - Zero extra dependencies
      - Splits on paragraph breaks + sentence boundaries via regex
      - Works well for plain text; fragile on abbreviations

    target_size:       max characters per chunk
    overlap_sentences: sentences of context to prepend from previous chunk (fallback only)
    min_chunk_size:    discard chunks shorter than this
    """
    def __init__(
        self,
        target_size: int = 600,
        overlap_sentences: int = 2,
        min_chunk_size: int = 80,
    ):
        self.target_size = target_size
        self.overlap_sentences = overlap_sentences
        self.min_chunk_size = min_chunk_size
        self._lib_splitter = None
        self._lib_md_splitter = None

        try:
            from semantic_text_splitter import TextSplitter, MarkdownSplitter
            # capacity as (min, max) to allow chunk merging up to target_size
            self._lib_splitter    = TextSplitter((min_chunk_size, target_size))
            self._lib_md_splitter = MarkdownSplitter((min_chunk_size, target_size))
            self._backend = "semantic-text-splitter (Rust)"
        except ImportError:
            self._backend = "regex (fallback — pip install semantic-text-splitter for better splits)"

    def split(self, text: str, doc_id: str, doc_name: str, metadata: dict) -> "List[Chunk]":
        if self._lib_splitter is not None:
            return self._split_lib(text, doc_id, doc_name, metadata)
        return self._split_regex(text, doc_id, doc_name, metadata)

    # ── Backend 1: semantic-text-splitter (Rust) ──────────────────────────────

    def _split_lib(self, text: str, doc_id: str, doc_name: str, metadata: dict) -> "List[Chunk]":
        # Use MarkdownSplitter if text has markdown-style headers
        splitter = (
            self._lib_md_splitter
            if re.search(r"^#{1,6}\s", text, re.MULTILINE)
            else self._lib_splitter
        )
        raw = splitter.chunks(text)
        return [
            Chunk(
                id=f"{doc_id}_{i}",
                doc_id=doc_id,
                doc_name=doc_name,
                text=c.strip(),
                metadata=metadata,
            )
            for i, c in enumerate(raw)
            if len(c.strip()) >= self.min_chunk_size
        ]

    # ── Backend 2: pure Python regex fallback ─────────────────────────────────

    def _split_regex(self, text: str, doc_id: str, doc_name: str, metadata: dict) -> "List[Chunk]":
        paragraphs = re.split(r"\n{2,}", text)
        _sent_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(\"'])")

        all_sentences: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            sents = _sent_end.split(para)
            all_sentences.extend(s.strip() for s in sents if s.strip())
            all_sentences.append("")  # paragraph boundary marker

        chunks: List[Chunk] = []
        current: List[str] = []
        current_size = 0
        idx = 0

        def _flush(sentences: List[str]) -> None:
            nonlocal idx
            text_out = " ".join(sentences).strip()
            if len(text_out) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=f"{doc_id}_{idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    text=text_out,
                    metadata=metadata,
                ))
                idx += 1

        for sent in all_sentences:
            if sent == "":  # paragraph boundary
                if current_size >= self.target_size * 0.4:
                    _flush(current)
                    current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                    current_size = sum(len(s) for s in current)
                continue
            current.append(sent)
            current_size += len(sent) + 1
            if current_size >= self.target_size:
                _flush(current)
                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current_size = sum(len(s) for s in current)

        if current:
            _flush(current)

        return chunks


def get_default_embedder():
    """Auto-detect best available embedder; fall back to BM25 (no dependencies)."""
    try:
        import openai
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbedder()
    except ImportError:
        pass
    try:
        import sentence_transformers
        return SentenceTransformerEmbedder()
    except ImportError:
        pass
    print("[RAG] No neural embedder found. Using BM25 (keyword search). "
          "For semantic search: pip install sentence-transformers")
    return BM25Retriever()


# ── Vector Store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Simple persistent vector store backed by a JSON/pickle file.
    No external DB needed. Swap for ChromaDB/Pinecone/pgvector in production.
    """

    def __init__(
        self,
        persist_dir: str = ".agent_memory/rag",
        embedder: Optional[BaseEmbedder] = None,
        chunker: Optional[SemanticChunker] = None,
        # Legacy params kept for backward compatibility (ignored when chunker is set)
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.persist_dir = persist_dir
        self.embedder = embedder or get_default_embedder()
        self.chunker = chunker or SemanticChunker(
            target_size=chunk_size,
            overlap_sentences=2,
        )
        self._chunks: List[Chunk] = []
        os.makedirs(persist_dir, exist_ok=True)
        self._load()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, doc_name: str, metadata: dict = None) -> int:
        """Chunk + embed + store raw text. Returns number of chunks added."""
        doc_id = hashlib.md5(f"{doc_name}{text[:100]}".encode()).hexdigest()[:8]

        # Remove existing chunks for same doc (re-ingest)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]

        chunks = self.chunker.split(text, doc_id, doc_name, metadata or {})

        if isinstance(self.embedder, BM25Retriever):
            # BM25 scores at query time — no embeddings needed
            self._chunks.extend(chunks)
        else:
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self._chunks.extend(chunks)

        self._save()
        return len(chunks)

    def ingest_file(self, path: str, metadata: dict = None) -> int:
        """Ingest a file. Supports .txt, .md, .py, .pdf (needs pypdf)."""
        ext = os.path.splitext(path)[-1].lower()
        doc_name = os.path.basename(path)

        if ext == ".pdf":
            text = self._read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        return self.ingest_text(text, doc_name, metadata)

    def ingest_url(self, url: str, metadata: dict = None) -> int:
        """Fetch a URL and ingest its text content."""
        import urllib.request
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode(errors="replace")
        # Strip HTML tags naively
        import re
        text = re.sub(r'<[^>]+>', ' ', raw)
        text = re.sub(r'\s+', ' ', text).strip()
        doc_name = url.split("/")[-1] or url
        return self.ingest_text(text, doc_name, metadata)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, doc_filter: str = None) -> List[Tuple[float, Chunk]]:
        """Return top_k most relevant chunks for a query, one per document (best chunk wins)."""
        if not self._chunks:
            return []

        if isinstance(self.embedder, BM25Retriever):
            candidates = [c for c in self._chunks
                          if not doc_filter or doc_filter in c.doc_name]
            if not candidates:
                return []
            self.embedder.fit([c.text for c in candidates])
            bm25_scores = self.embedder.get_scores(query)
            all_scored = sorted(zip(bm25_scores, candidates), key=lambda x: -x[0])
        else:
            q_emb = self.embedder.embed([query])[0]
            all_scored = []
            for chunk in self._chunks:
                if doc_filter and doc_filter not in chunk.doc_name:
                    continue
                if chunk.embedding is None:
                    continue
                score = self._cosine(q_emb, chunk.embedding)
                all_scored.append((score, chunk))
            all_scored.sort(key=lambda x: -x[0])

        # Deduplicate: keep only the highest-scoring chunk per document
        seen_docs: set = set()
        deduped = []
        for score, chunk in all_scored:
            if chunk.doc_id not in seen_docs:
                seen_docs.add(chunk.doc_id)
                deduped.append((score, chunk))
            if len(deduped) == top_k:
                break

        return deduped

    def format_results(self, results: List[Tuple[float, Chunk]]) -> str:
        """Format search results into a context string for the LLM."""
        if not results:
            return "No relevant documents found."
        parts = []
        for i, (score, chunk) in enumerate(results, 1):
            parts.append(
                f"[{i}] Source: {chunk.doc_name} (relevance: {score:.2f})\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def list_documents(self) -> List[dict]:
        seen = {}
        for c in self._chunks:
            if c.doc_id not in seen:
                seen[c.doc_id] = {"doc_id": c.doc_id, "doc_name": c.doc_name, "chunks": 0}
            seen[c.doc_id]["chunks"] += 1
        return list(seen.values())

    def clear(self):
        self._chunks = []
        self._save()

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("Install pypdf to read PDFs: pip install pypdf")

    def _save(self):
        path = os.path.join(self.persist_dir, "store.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._chunks, f)

    def _load(self):
        path = os.path.join(self.persist_dir, "store.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._chunks = pickle.load(f)
            except Exception:
                self._chunks = []


# ── Register as agent tools ───────────────────────────────────────────────────

def register_rag_tools(registry: ToolRegistry, store: VectorStore):
    """
    Register RAG tools onto your ToolRegistry.
    The agent can now call these tools just like any other tool.
    """

    @registry.tool(
        description=(
            "Search the knowledge base for information relevant to a query. "
            "Returns the most relevant document chunks. "
            "Use this before answering any question that might be in the documents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                "doc_filter": {"type": "string", "description": "Filter by document name (optional)"},
            },
            "required": ["query"],
        },
        tags=["rag", "knowledge"],
    )
    def search_knowledge_base(query: str, top_k: int = 5, doc_filter: str = None) -> str:
        results = store.search(query, top_k=top_k, doc_filter=doc_filter)
        return store.format_results(results)

    @registry.tool(
        description="Ingest a file or URL into the knowledge base so it can be searched.",
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "File path or URL to ingest"},
                "metadata": {"type": "object", "description": "Optional metadata tags", "default": {}},
            },
            "required": ["source"],
        },
        tags=["rag", "knowledge"],
    )
    def ingest_document(source: str, metadata: dict = {}) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            n = store.ingest_url(source, metadata)
        else:
            n = store.ingest_file(source, metadata)
        return f"Ingested {n} chunks from: {source}"

    @registry.tool(
        description="Add raw text directly to the knowledge base.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text content to store"},
                "doc_name": {"type": "string", "description": "A name/label for this document"},
                "metadata": {"type": "object", "default": {}},
            },
            "required": ["text", "doc_name"],
        },
        tags=["rag", "knowledge"],
    )
    def ingest_text(text: str, doc_name: str, metadata: dict = {}) -> str:
        n = store.ingest_text(text, doc_name, metadata)
        return f"Ingested {n} chunks as '{doc_name}'"

    @registry.tool(
        description="List all documents currently in the knowledge base.",
        input_schema={"type": "object", "properties": {}, "required": []},
        tags=["rag", "knowledge"],
    )
    def list_documents() -> str:
        docs = store.list_documents()
        if not docs:
            return "Knowledge base is empty. Use ingest_document to add documents."
        lines = [f"  {d['doc_name']}  ({d['chunks']} chunks)" for d in docs]
        return f"{len(docs)} documents in knowledge base:\n" + "\n".join(lines)
