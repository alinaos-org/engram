"""Stage 5a: token-aware chunking at paragraph boundaries (research/phase1-corpus.md §b).

Each document gets one write size, drawn (seeded by source and document id) from ``sizes``. A cut
snaps to the paragraph boundary nearest the target within +-``tolerance``; failing that, to the
nearest line or sentence boundary; failing that (one enormous sentence), to a word boundary.
Paragraphs are separated by blank lines. Conversation sources (``mode="session"``) write one
chunk per session (paragraph), splitting only sessions longer than ``session_max`` tokens.

Every chunk keeps ``char_offsets`` into its document, so ``text[start:end]`` is verbatim and probes
with document-level spans can be mapped onto chunks.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from engram.corpus.io import sha256
from engram.corpus.tokenize import Tokenizer

_PARA_SEP = re.compile(r"\n[ \t\r\f\v]*\n\s*")
_SENT_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+")


@dataclass
class Segment:
    start: int
    end: int
    n_tokens: int
    kind: str  # para (ends a paragraph) | line | sentence | word


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraphs(text: str) -> Iterator[tuple[int, int]]:
    pos = 0
    for m in _PARA_SEP.finditer(text):
        s, e = _strip(text, pos, m.start())
        if e > s:
            yield s, e
        pos = m.end()
    s, e = _strip(text, pos, len(text))
    if e > s:
        yield s, e


def _split(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    out, pos = [], start
    for m in pattern.finditer(text, start, end):
        s, e = _strip(text, pos, m.start())
        if e > s:
            out.append((s, e))
        pos = m.end()
    s, e = _strip(text, pos, end)
    if e > s:
        out.append((s, e))
    return out


def _words(text: str, start: int, end: int, tok: Tokenizer, limit: int) -> list[Segment]:
    out: list[Segment] = []
    cur_s, cur_n, last_e = None, 0, start
    for m in re.finditer(r"\S+", text[start:end]):
        s, e = start + m.start(), start + m.end()
        n = tok.count(m.group())
        if cur_s is not None and cur_n + n > limit:
            out.append(Segment(cur_s, last_e, cur_n, "word"))
            cur_s, cur_n = None, 0
        if cur_s is None:
            cur_s = s
        cur_n += n
        last_e = e
    if cur_s is not None:
        out.append(Segment(cur_s, last_e, cur_n, "word"))
    return out


def segments(text: str, tok: Tokenizer, limit: int) -> list[Segment]:
    """Paragraphs no longer than ``limit`` tokens; longer ones split by line, sentence, word."""
    out: list[Segment] = []
    for ps, pe in paragraphs(text):
        n = tok.count(text[ps:pe])
        if n <= limit:
            out.append(Segment(ps, pe, n, "para"))
            continue
        pieces: list[Segment] = []
        for ls, le in _split(text, ps, pe, re.compile(r"\n")):
            ln = tok.count(text[ls:le])
            if ln <= limit:
                pieces.append(Segment(ls, le, ln, "line"))
                continue
            for ss, se in _split(text, ls, le, _SENT_END):
                sn = tok.count(text[ss:se])
                if sn <= limit:
                    pieces.append(Segment(ss, se, sn, "sentence"))
                else:
                    pieces.extend(_words(text, ss, se, tok, limit))
        pieces[-1].kind = "para"
        out.extend(pieces)
    return out


def cut(segs: list[Segment], target: int, tolerance: float = 0.1) -> list[tuple[int, int]]:
    """Group segments into chunks; returns (first, last) segment index pairs."""
    lo, hi = target * (1 - tolerance), target * (1 + tolerance)
    groups = []
    i = 0
    while i < len(segs):
        total = 0
        best: tuple[tuple[int, float], int] | None = None
        for j in range(i, len(segs)):
            total += segs[j].n_tokens
            if total > hi and j > i:
                break
            in_band = segs[j].kind == "para" and lo <= total <= hi
            rank = (0 if in_band else 1, abs(total - target))
            if best is None or rank < best[0]:
                best = (rank, j)
        assert best is not None
        groups.append((i, best[1]))
        i = best[1] + 1
    return groups


def assign_split(source: str, doc_id: str, val: float = 0.01, test: float = 0.01) -> str:
    """Split by document, never by chunk: a stable hash of (source, doc_id)."""
    u = int(sha256(f"{source}\x00{doc_id}")[:12], 16) / 16**12
    if u < test:
        return "test"
    if u < test + val:
        return "val"
    return "train"


def chunk_doc(
    doc: dict[str, Any],
    tok: Tokenizer,
    *,
    sizes: list[int],
    seed: int = 0,
    tolerance: float = 0.1,
    mode: str = "tokens",
    session_max: int = 4096,
    split: str | None = None,
    slice_: str | None = None,
) -> list[dict[str, Any]]:
    text = doc["text"]
    if mode == "session":
        size: int | str = "session"
        segs = segments(text, tok, session_max)
        groups = []
        start = 0
        for k, s in enumerate(segs):  # one chunk per paragraph unless a paragraph was split
            if s.kind == "para":
                sub = segs[start : k + 1]
                if len(sub) == 1:
                    groups.append((start, k))
                else:
                    groups.extend((start + a, start + b) for a, b in cut(sub, session_max, 0.0))
                start = k + 1
    else:
        rng = random.Random(f"chunk:{seed}:{doc['source']}:{doc['doc_id']}")
        size = rng.choice(sizes)
        segs = segments(text, tok, int(size * (1 + tolerance)))
        groups = cut(segs, size, tolerance)
    split = split or doc.get("force_split") or assign_split(doc["source"], doc["doc_id"])
    out = []
    for idx, (a, b) in enumerate(groups):
        s, e = segs[a].start, segs[b].end
        body = text[s:e]
        out.append(
            {
                "id": f"{doc['source']}/{doc['doc_id']}/{size}/{idx:05d}",
                "source": doc["source"],
                "doc_id": doc["doc_id"],
                "licence": doc.get("licence", ""),
                "slice": slice_,
                "split": split,
                "chunk_index": idx,
                "chunk_size": size,
                "char_offsets": [s, e],
                "n_tokens": tok.count(body),
                "boundary": segs[b].kind,
                "sha256": sha256(body),
                "text": body,
            }
        )
    return out


def chunk_docs(
    docs: Iterable[dict[str, Any]],
    tok: Tokenizer,
    *,
    sizes: list[int],
    seed: int = 0,
    tolerance: float = 0.1,
    modes: dict[str, str] | None = None,
    slices: dict[str, str | None] | None = None,
    session_max: int = 4096,
    val: float = 0.01,
    test: float = 0.01,
) -> Iterator[dict[str, Any]]:
    modes = modes or {}
    slices = slices or {}
    for doc in docs:
        split = doc.get("force_split") or assign_split(doc["source"], doc["doc_id"], val, test)
        yield from chunk_doc(
            doc,
            tok,
            sizes=sizes,
            seed=seed,
            tolerance=tolerance,
            mode=modes.get(doc["source"], "tokens"),
            session_max=session_max,
            split=split,
            slice_=slices.get(doc["source"]),
        )
