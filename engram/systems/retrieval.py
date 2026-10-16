"""A small BM25 retriever with no dependencies, used by the retrieval baselines."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
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
    """BM25 with per-owner corpus statistics.

    With ``owner`` set, both the eligible documents *and* the term statistics (document
    frequencies, average length) come from that owner's documents only. Pooled statistics would
    let other users' data change one user's ranking, a small cross-user side channel.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[Doc] = []
        self._tf: list[Counter] = []
        self._df: dict[str | None, Counter] = defaultdict(Counter)
        self._n: Counter = Counter()
        self._len: Counter = Counter()

    def add(self, doc: Doc) -> None:
        tf = Counter(tokenize(doc.text))
        self.docs.append(doc)
        self._tf.append(tf)
        size = sum(tf.values())
        for key in (None, doc.owner):
            self._df[key].update(tf.keys())
            self._n[key] += 1
            self._len[key] += size

    def remove(self, pred) -> int:
        """Drop every document matching ``pred`` and rebuild statistics."""
        keep = [d for d in self.docs if not pred(d)]
        removed = len(self.docs) - len(keep)
        self.docs, self._tf = [], []
        self._df, self._n, self._len = defaultdict(Counter), Counter(), Counter()
        for d in keep:
            self.add(d)
        return removed

    def search(self, query: str, k: int = 5, owner: str | None = None) -> list[Doc]:
        """Top-k documents. With ``owner`` set, only that owner's documents are eligible."""
        q = tokenize(query)
        n = self._n[owner]
        if not n or not q:
            return []
        df = self._df[owner]
        avgdl = self._len[owner] / n
        scored = []
        for i, (doc, tf) in enumerate(zip(self.docs, self._tf, strict=True)):
            if owner is not None and doc.owner != owner:
                continue
            dl = sum(tf.values())
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                f = tf[term]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.docs[i] for _, i in scored[:k]]
