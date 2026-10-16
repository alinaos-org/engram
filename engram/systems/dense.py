"""Dense and hybrid retrieval, so the retrieval baselines are not lexical strawmen.

BM25 misses paraphrases: "Which editor do I use?" never matches "I do everything in Helix". The
frontier+RAG baseline (B2) is the claim Engram has to beat, so it gets a fair retriever: dense
embeddings fused with BM25 by reciprocal rank fusion. Needs the ``ml`` extra.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

from engram.systems.retrieval import BM25, Doc

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Thin, thread-safe wrapper around a sentence-transformers model (loaded lazily, shared)."""

    _cache: dict[str, Embedder] = {}
    _cache_lock = threading.Lock()

    def __init__(self, model_id: str = DEFAULT_EMBEDDER, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id, device=device)
        self._lock = threading.Lock()

    @classmethod
    def shared(cls, model_id: str = DEFAULT_EMBEDDER) -> Embedder:
        with cls._cache_lock:
            if model_id not in cls._cache:
                cls._cache[model_id] = cls(model_id)
            return cls._cache[model_id]

    def encode(self, texts: list[str]):
        with self._lock:
            return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


@dataclass
class DenseIndex:
    embedder: Embedder
    docs: list[Doc] = field(default_factory=list)
    _vecs: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    _rows: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add(self, doc: Doc) -> None:
        vec = self.embedder.encode([doc.text])[0]
        self._rows[doc.owner].append(len(self.docs))
        self._vecs[doc.owner].append(vec)
        self.docs.append(doc)

    def remove(self, pred) -> int:
        keep = [d for d in self.docs if not pred(d)]
        removed = len(self.docs) - len(keep)
        self.docs, self._vecs, self._rows = [], defaultdict(list), defaultdict(list)
        for d in keep:
            self.add(d)
        return removed

    def search(self, query: str, k: int = 5, owner: str | None = None) -> list[Doc]:
        import numpy as np

        owners = [owner] if owner is not None else list(self._vecs)
        rows = [r for o in owners for r in self._rows.get(o, [])]
        if not rows:
            return []
        mat = np.stack([v for o in owners for v in self._vecs.get(o, [])])
        q = self.embedder.encode([query])[0]
        sims = mat @ q
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.docs[rows[i]] for i in order]


class HybridIndex:
    """BM25 + dense, fused by reciprocal rank fusion (k=60)."""

    def __init__(self, embedder: Embedder, depth: int = 20, rrf_k: int = 60) -> None:
        self.bm25 = BM25()
        self.dense = DenseIndex(embedder)
        self.depth, self.rrf_k = depth, rrf_k

    @property
    def docs(self) -> list[Doc]:
        return self.bm25.docs

    def add(self, doc: Doc) -> None:
        self.bm25.add(doc)
        self.dense.add(doc)

    def remove(self, pred) -> int:
        self.dense.remove(pred)
        return self.bm25.remove(pred)

    def search(self, query: str, k: int = 5, owner: str | None = None) -> list[Doc]:
        scores: dict[int, float] = defaultdict(float)
        first_seen: dict[int, Doc] = {}
        for ranked in (
            self.bm25.search(query, self.depth, owner),
            self.dense.search(query, self.depth, owner),
        ):
            for rank, doc in enumerate(ranked):
                scores[id(doc)] += 1.0 / (self.rrf_k + rank + 1)
                first_seen.setdefault(id(doc), doc)
        best = sorted(scores, key=lambda d: -scores[d])[:k]
        return [first_seen[d] for d in best]
