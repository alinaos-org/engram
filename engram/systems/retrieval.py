"""A small BM25 retriever with no dependencies, used by the retrieval baselines."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Doc:
    text: str
    owner: str
    meta: dict = field(default_factory=dict)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[Doc] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()

    def add(self, doc: Doc) -> None:
        tf = Counter(tokenize(doc.text))
        self.docs.append(doc)
        self._tf.append(tf)
        self._df.update(tf.keys())

    def remove(self, pred) -> int:
        """Drop every document matching ``pred`` and rebuild statistics."""
        keep = [d for d in self.docs if not pred(d)]
        removed = len(self.docs) - len(keep)
        self.docs, self._tf, self._df = [], [], Counter()
        for d in keep:
            self.add(d)
        return removed

    def search(self, query: str, k: int = 5, owner: str | None = None) -> list[Doc]:
        """Top-k documents. With ``owner`` set, only that owner's documents are eligible."""
        q = tokenize(query)
        if not self.docs or not q:
            return []
        n = len(self.docs)
        avgdl = sum(sum(tf.values()) for tf in self._tf) / n
        scored = []
        for i, (doc, tf) in enumerate(zip(self.docs, self._tf, strict=True)):
            if owner is not None and doc.owner != owner:
                continue
            dl = sum(tf.values())
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                f = tf[term]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.docs[i] for _, i in scored[:k]]
