"""Stage 2: document filters (research/phase1-corpus.md §e.2 and §d.2).

Per document, in order: hold-outs (PG-19 val/test Gutenberg ids, denylisted arXiv ids), language,
per-document licence allowlist, length (at least ``min_chunks`` chunks), OCR quality, biography
heuristic, then MinHash near-duplicate removal across all sources (earlier sources win).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from engram.corpus import sources as src
from engram.corpus.text import MinHasher, count_words, sample_rate
from engram.corpus.tokenize import Tokenizer

_STOP = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "a",
        "in",
        "is",
        "that",
        "it",
        "was",
        "for",
        "on",
        "as",
        "with",
        "by",
        "he",
        "she",
        "they",
        "be",
        "at",
        "from",
        "his",
        "her",
        "this",
        "which",
        "or",
        "an",
        "are",
        "not",
        "but",
        "had",
        "have",
        "were",
        "you",
        "i",
        "we",
    ]
)
_GOOD_WORD = re.compile(r"^[\"'(\[“‘]*[A-Za-z][a-z'’-]*[.,;:!?\"')\]”’]*$|^\d+[.,;:]?$")
_BIO_PATTERNS = [
    re.compile(r"\(\s*(?:née\s+\S+\s*[;,]\s*)?born\b", re.I),
    re.compile(r"\(\s*[^)]{0,80}?\b\d{3,4}\s*[–—-]\s*(?:\d{1,2}\s+\w+\s+)?\d{3,4}\s*\)"),
    re.compile(r"\(\s*(?:c\.\s*)?\d{1,2}\s+[A-Z][a-z]+\s+\d{4}"),
    re.compile(r"\(\s*[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"),
    re.compile(
        r"\b(?:is|was) an? (?:\w+ ){0,3}(?:born|politician|actor|actress|singer|footballer"
        r"|writer|author|musician|businessman|businesswoman|athlete|player)\b"
    ),
]
_GUTENBERG_ID = re.compile(r"gutenberg\.org/(?:ebooks|files|cache/epub)/(\d+)")


def english_score(text: str, limit: int = 2000) -> float:
    toks = text.lower().split()[:limit]
    if not toks:
        return 0.0
    return sum(t.strip(".,;:!?\"'()") in _STOP for t in toks) / len(toks)


def ocr_score(text: str, limit: int = 5000) -> float:
    toks = text.split()[:limit]
    if not toks:
        return 0.0
    return sum(bool(_GOOD_WORD.match(t)) for t in toks) / len(toks)


def looks_biographical(doc: dict[str, Any]) -> bool:
    lead = (doc.get("meta") or {}).get("lead") or doc["text"][:400]
    first = re.split(r"(?<=[.!?])\s", lead.strip(), maxsplit=1)[0]
    return any(p.search(first) for p in _BIO_PATTERNS)


def gutenberg_id(doc: dict[str, Any]) -> str | None:
    meta = doc.get("meta") or {}
    for value in (meta.get("url"), meta.get("source"), meta.get("gutenberg_id")):
        if value:
            m = _GUTENBERG_ID.search(str(value))
            if m:
                return m.group(1)
            if str(value).isdigit():
                return str(value)
    if doc["source"] in ("pg19", "gutenberg") and str(doc["doc_id"]).isdigit():
        return str(doc["doc_id"])
    return None


def holdout_ids(docs: Iterable[dict[str, Any]]) -> set[str]:
    """Gutenberg ids of PG-19 validation/test books (fetched with ``force_split``)."""
    out = set()
    for d in docs:
        if d["source"] == "pg19" and d.get("force_split") in ("val", "test"):
            gid = gutenberg_id(d)
            if gid:
                out.add(gid)
    return out


def _mentions_denylisted_arxiv(doc: dict[str, Any], denylist: Iterable[str]) -> str | None:
    hay = " ".join([str(doc["doc_id"]), *(str(v) for v in (doc.get("meta") or {}).values())])
    for aid in denylist:
        if aid in hay:
            return aid
    return None


class DocFilter:
    """Streaming document filter: feed sources in order, earlier sources win near-duplicates.

    Memory holds only the hold-out ids, MinHash signatures and counters, never the documents.
    """

    def __init__(
        self, cfg: dict[str, Any], tokenizer: Tokenizer, held: set[str] | None = None
    ) -> None:
        self.min_chunks = int(cfg.get("min_chunks", 3))
        self.min_tokens = self.min_chunks * int(cfg.get("min_chunk_tokens", 1024))
        self.min_english = float(cfg.get("min_english_score", 0.08))
        self.min_ocr = float(cfg.get("min_ocr_score", 0.8))
        self.allow = set(cfg.get("licence_allowlist", src.LICENCE_ALLOWLIST))
        self.denylist = tuple(cfg.get("arxiv_denylist", src.ARXIV_DENYLIST))
        self.dedup_threshold = float(cfg.get("dedup_threshold", 0.8))
        self.tok = tokenizer
        self.held = held or set()
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._mh = MinHasher()
        self._buckets: dict[tuple[int, tuple[int, ...]], list[tuple[int, ...]]] = defaultdict(list)

    def reason(self, d: dict[str, Any], spec: dict[str, Any]) -> str | None:
        filters = set(spec.get("filters") or ())
        if d.get("force_split"):
            return None  # held-out eval books are kept as-is
        if "pg19_holdout" in filters and gutenberg_id(d) in self.held:
            return "pg19_holdout"
        if "arxiv_denylist" in filters and _mentions_denylisted_arxiv(d, self.denylist):
            return "arxiv_denylist"
        if not spec.get("generated") and english_score(d["text"]) < self.min_english:
            return "language"
        if spec.get("licence_field") and src.normalise_licence(d.get("licence")) not in self.allow:
            return "licence"
        if "ocr" in filters and ocr_score(d["text"]) < self.min_ocr:
            return "ocr"
        if "biography" in filters and looks_biographical(d):
            return "biography"
        if spec.get("chunk_mode") == "session":
            paras = [p for p in re.split(r"\n\s*\n", d["text"]) if p.strip()]
            return "length" if len(paras) < self.min_chunks else None
        if self.tok.count(d["text"]) < self.min_tokens:
            return "length"
        return None

    def is_duplicate(self, text: str) -> bool:
        if self.dedup_threshold >= 1.0:
            return False
        mh = self._mh
        sig = mh.signature(text, sample=sample_rate(count_words(text)))
        for key in mh.band_keys(sig):
            if any(mh.similarity(sig, o) >= self.dedup_threshold for o in self._buckets[key]):
                return True
        for key in mh.band_keys(sig):
            self._buckets[key].append(sig)
        return False

    def apply(
        self, name: str, spec: dict[str, Any], docs: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        c = self.counts[name]
        for d in docs:
            c["in"] += 1
            why = self.reason(d, spec)
            if why is None and self.is_duplicate(d["text"]):
                why = "near_duplicate"
            if why:
                c[f"dropped:{why}"] += 1
                continue
            c["kept"] += 1
            yield d

    def report(self) -> dict[str, Any]:
        return {
            "min_tokens": self.min_tokens,
            "holdout_gutenberg_ids": sorted(self.held),
            "sources": {
                name: {
                    "in": c["in"],
                    "kept": c["kept"],
                    "dropped": {k.split(":", 1)[1]: v for k, v in sorted(c.items()) if ":" in k},
                }
                for name, c in self.counts.items()
            },
        }


def filter_docs(
    docs_by_source: dict[str, list[dict[str, Any]]],
    specs: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    tokenizer: Tokenizer,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """In-memory convenience wrapper around ``DocFilter``."""
    held = holdout_ids(d for docs in docs_by_source.values() for d in docs)
    f = DocFilter(cfg, tokenizer, held)
    kept = {name: list(f.apply(name, specs[name], docs)) for name, docs in docs_by_source.items()}
    return kept, f.report()
