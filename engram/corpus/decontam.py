"""Stage 3: decontamination against the eval sets (research/phase1-corpus.md §d).

The reference set R holds, per reference source (engram-bench, LongMemEval, LoCoMo, PrefEval):

* 13-word shingles of every text (§d.4), minus stock phrases that occur in more than
  ``max_df`` of a background sample (only when the sample has at least ``min_docs`` documents);
* engram-bench templates compiled into regexes with slot wildcards (§d.5);
* engram-bench canary strings and generated user names (§d.6);
* eval questions, for the probe-level near-duplicate filter (§d.7, Jaccard on word 5-grams).

A chunk is dropped on any hit. A canary or user-name hit anywhere, or a template hit in generated
data, is *fatal*: ``report["fatal"]`` lists it and the build fails (§d.5, §d.6).

Missing reference files are skipped and listed in the report as ``absent``. Shingle sets use a
sorted ``numpy`` array when numpy is importable (8 bytes per shingle) and a Python set otherwise.
"""

from __future__ import annotations

import json
import random
import re
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram.corpus.text import hash64, jaccard, shingle_set, shingles, words

SENT_V = "\x00V\x00"
SENT_OLD = "\x00O\x00"
_WILDCARD = r"[^.!?\n]{1,60}?"
MIN_ANCHOR = 15

DEFAULT_EXTERNAL = [
    "data/external/longmemeval_oracle.json",
    "data/external/longmemeval_s_cleaned.json",
    "data/external/longmemeval_m_cleaned.json",
    "data/external/locomo10.json",
    "data/external/prefeval",
]


class ShingleIndex:
    """A frozen set of 64-bit shingle hashes."""

    def __init__(self) -> None:
        self._buf: array | None = array("Q")
        self._np: Any = None
        self._set: set[int] | None = None
        self._removed: set[int] = set()

    def add(self, hashes: Iterable[int]) -> None:
        assert self._buf is not None, "index is frozen"
        self._buf.extend(hashes)

    def freeze(self) -> ShingleIndex:
        if self._buf is None:
            return self
        try:
            import numpy as np

            self._np = np.unique(np.frombuffer(self._buf, dtype=np.uint64))
        except ImportError:
            self._set = set(self._buf)
        self._buf = None
        return self

    def __len__(self) -> int:
        self.freeze()
        n = len(self._np) if self._np is not None else len(self._set or ())
        return n - len(self._removed)

    def hits(self, hashes: list[int]) -> list[int]:
        self.freeze()
        if not hashes:
            return []
        if self._np is not None:
            import numpy as np

            q = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
            if not len(self._np):
                return []
            idx = np.searchsorted(self._np, q).clip(max=len(self._np) - 1)
            found = [int(h) for h in q[self._np[idx] == q]]
        else:
            ref = self._set or set()
            found = [h for h in hashes if h in ref]
        return [h for h in found if h not in self._removed]

    def remove(self, hashes: Iterable[int]) -> None:
        self._removed.update(hashes)


@dataclass
class Template:
    origin: str
    text: str
    anchor: str
    regex: re.Pattern[str]


def compile_template(origin: str, text: str) -> Template | None:
    norm = " ".join(text.split())
    parts = re.split(f"({re.escape(SENT_V)}|{re.escape(SENT_OLD)})", norm)
    literals = [p for p in parts if p not in (SENT_V, SENT_OLD)]
    anchor = max((p.strip().lower() for p in literals), key=len, default="")
    if len(anchor) < MIN_ANCHOR:
        return None
    pattern = "".join(
        _WILDCARD if p in (SENT_V, SENT_OLD) else re.escape(p).replace("\\ ", " ") for p in parts
    )
    return Template(origin, norm, anchor, re.compile(pattern, re.I))


def bench_template_strings() -> list[tuple[str, str]]:
    """Every engram-bench template string, with sentinels where values go."""
    from engram.bench import world

    out: list[tuple[str, str]] = []
    for slot in world.SLOTS:
        out += [(f"world.{slot.name}.assert", t.format(v=SENT_V)) for t in slot.assert_t]
        out += [
            (f"world.{slot.name}.update", t.format(v=SENT_V, old=SENT_OLD)) for t in slot.update_t
        ]
        out.append((f"world.{slot.name}.question", slot.question))
    try:
        from engram.bench import writer

        out += [("writer.FILLER", f) for f in getattr(writer, "FILLER", [])]
        out += _rendered_writer_templates(writer, world)
    except Exception as e:  # noqa: BLE001 - writer.py is optional for decontam
        out.append(("writer.unavailable", repr(e)))
    try:
        from engram.bench import probes

        for slot, (question, _) in getattr(probes, "APPLICATION", {}).items():
            out.append((f"probes.application.{slot}", question))
    except Exception:  # noqa: BLE001
        pass
    return [(o, t) for o, t in out if o != "writer.unavailable"]


def _rendered_writer_templates(writer: Any, world: Any) -> list[tuple[str, str]]:
    """Render writer.template_text for each event kind with sentinel values."""
    from engram.bench.schema import Event, EventKind, MemoryItem, User

    items = [MemoryItem(s.name, "u", s.name, s.type, s.topic) for s in world.SLOTS]
    user = User("u", "u", "u", items=items)
    rng = random.Random(0)
    out = []
    events = [Event(EventKind.CANARY, 0, value=SENT_V), Event(EventKind.FORGET_ALL, 0)]
    events += [
        Event(EventKind.FORGET_TOPIC, 0, topic=t) for t in getattr(writer, "TOPIC_PHRASE", {})
    ]
    events += [Event(EventKind.FORGET_ITEM, 0, item_id=s.name) for s in world.SLOTS]
    for ev in events:
        try:
            out.append((f"writer.{ev.kind.value}", writer.template_text(ev, user, rng)))
        except Exception:  # noqa: BLE001 - signature drift in writer.py: skip that template
            continue
    return out


def _strings(obj: Any, min_words: int) -> Iterator[str]:
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if len(x.split()) >= min_words:
                yield x
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)


def _questions(obj: Any) -> Iterator[str]:
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "question" and isinstance(v, str):
                    yield v
                else:
                    stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)


def _load_json_any(path: Path) -> Iterator[Any]:
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ReferenceSet:
    n: int = 13
    indices: dict[str, ShingleIndex] = field(default_factory=dict)
    templates: list[Template] = field(default_factory=list)
    canaries: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    questions: list[tuple[str, set[int]]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    stock_removed: dict[str, int] = field(default_factory=dict)
    _qindex: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    _qseen: set[str] = field(default_factory=set)
    _sweep: re.Pattern[str] | None = None

    # -- building ---------------------------------------------------------------------------

    def add_texts(self, origin: str, texts: Iterable[str]) -> int:
        idx = self.indices.setdefault(origin, ShingleIndex())
        n = 0
        for t in set(texts):
            idx.add(shingles(words(t), self.n))
            n += 1
        return n

    def add_questions(self, origin: str, qs: Iterable[str]) -> int:
        n = 0
        for q in sorted(set(qs) - self._qseen):
            self._qseen.add(q)
            sh = shingle_set(q, 5, short_ok=True)
            if not sh:
                continue
            i = len(self.questions)
            self.questions.append((f"{origin}: {q}", sh))
            for h in sh:
                self._qindex[h].append(i)
            n += 1
        return n

    def add_bench_users(self, origin: str, users: Iterable[dict[str, Any]]) -> None:
        texts, qs = [], []
        for u in users:
            self.canaries.add(u["canary"])
            self.names.add(u["name"])
            for s in u.get("sessions", []):
                turns = [t["text"] for t in s.get("turns", [])]
                texts.append("\n".join(turns))
                texts.extend(turns)
            for p in u.get("probes", []):
                qs.append(p["question"])
        n = self.add_texts(origin, texts)
        nq = self.add_questions(origin, qs)
        self.sources.append({"name": origin, "status": "loaded", "texts": n, "questions": nq})

    def add_external(self, path: Path) -> None:
        files = sorted(path.rglob("*.json*")) if path.is_dir() else [path]
        if not path.exists() or not files:
            self.sources.append({"name": str(path), "status": "absent"})
            return
        for f in files:
            texts: list[str] = []
            qs: list[str] = []
            for obj in _load_json_any(f):
                texts.extend(_strings(obj, self.n))
                qs.extend(_questions(obj))
            n = self.add_texts(str(f), texts)
            nq = self.add_questions(str(f), qs)
            self.sources.append({"name": str(f), "status": "loaded", "texts": n, "questions": nq})

    def add_templates(self, strings: Iterable[tuple[str, str]]) -> None:
        seen = {t.text for t in self.templates}
        for origin, s in strings:
            t = compile_template(origin, s)
            if t and t.text not in seen:
                seen.add(t.text)
                self.templates.append(t)

    def freeze(self) -> ReferenceSet:
        for idx in self.indices.values():
            idx.freeze()
        sweep = sorted(self.canaries | self.names, key=len, reverse=True)
        if sweep:
            alt = "|".join(re.escape(s) for s in sweep)
            self._sweep = re.compile(rf"(?<!\w)(?:{alt})(?!\w)")
        for s in self.sources:
            if s.get("status") == "loaded" and s["name"] in self.indices:
                s["shingles"] = len(self.indices[s["name"]])
        return self

    def drop_stock_phrases(
        self, background: Iterable[str], *, max_df: float = 0.001, min_docs: int = 1000
    ) -> dict[str, Any]:
        """Remove reference shingles found in more than ``max_df`` of the background docs."""
        df: dict[str, Counter[int]] = {k: Counter() for k in self.indices}
        n_docs = 0
        for text in background:
            n_docs += 1
            hs = list(set(shingles(words(text), self.n)))
            for origin, idx in self.indices.items():
                df[origin].update(idx.hits(hs))
        if n_docs < min_docs:
            return {"status": "skipped", "background_docs": n_docs, "min_docs": min_docs}
        limit = max_df * n_docs
        for origin, idx in self.indices.items():
            stock = [h for h, c in df[origin].items() if c > limit]
            idx.remove(stock)
            self.stock_removed[origin] = len(stock)
        return {"status": "applied", "background_docs": n_docs, "removed": self.stock_removed}

    # -- checking ---------------------------------------------------------------------------

    def check_text(self, text: str, *, ngrams: bool = True) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        norm = " ".join(text.split())
        if self._sweep is not None:
            for m in self._sweep.finditer(norm):
                kind = "canary" if m.group() in self.canaries else "name"
                hits.append({"kind": kind, "ref": "engram-bench", "match": m.group()})
        low = norm.lower()
        for t in self.templates:
            if t.anchor in low:
                m = t.regex.search(norm)
                if m:
                    hits.append({"kind": "template", "ref": t.origin, "match": m.group()})
        if ngrams and self.indices:
            toks = words(text)
            grams = [
                (i, hash64(" ".join(toks[i : i + self.n]))) for i in range(len(toks) - self.n + 1)
            ]
            hs = [h for _, h in grams]
            for origin, idx in self.indices.items():
                found = set(idx.hits(hs))
                if found:
                    i = next(i for i, h in grams if h in found)
                    hits.append(
                        {
                            "kind": "ngram",
                            "ref": origin,
                            "count": str(len(found)),
                            "match": " ".join(toks[i : i + self.n]),
                        }
                    )
        return hits

    def near_duplicate_question(self, question: str, threshold: float = 0.8) -> str | None:
        sh = shingle_set(question, 5, short_ok=True)
        cands = {i for h in sh for i in self._qindex.get(h, ())}
        for i in sorted(cands):
            if jaccard(sh, self.questions[i][1]) >= threshold:
                return self.questions[i][0]
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "sources": self.sources,
            "templates": len(self.templates),
            "canaries": len(self.canaries),
            "user_names": len(self.names),
            "eval_questions": len(self.questions),
            "stock_removed": self.stock_removed,
        }


def build_reference(
    *,
    n: int = 13,
    bench: dict[str, Any] | None = None,
    bench_dirs: Iterable[str | Path] = (),
    external: Iterable[str | Path] = (),
) -> ReferenceSet:
    """Assemble R. ``bench`` is a BenchConfig dict to generate engram-bench v0 (None = skip)."""
    ref = ReferenceSet(n=n)
    ref.add_templates(bench_template_strings())
    if bench is not None:
        from engram.bench import dataset
        from engram.bench.schema import to_json

        cfg = dataset.BenchConfig(**bench)
        b = dataset.generate(cfg)
        ref.add_bench_users(f"engram-bench(seed={cfg.seed})", (to_json(u) for u in b.users))
    for d in bench_dirs:
        users = Path(d) / "users.jsonl"
        if not users.exists():
            ref.sources.append({"name": str(d), "status": "absent"})
            continue
        with open(users, encoding="utf-8", newline="\n") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ref.add_bench_users(str(d), rows)
    for p in external:
        ref.add_external(Path(p))
    return ref.freeze()


def _fatal(hit: dict[str, str], generated: bool) -> bool:
    return hit["kind"] in ("canary", "name") or (generated and hit["kind"] == "template")


def iter_clean_chunks(
    chunks: Iterable[dict[str, Any]],
    ref: ReferenceSet,
    report: dict[str, Any],
    *,
    generated_sources: set[str] = frozenset(),  # type: ignore[assignment]
    examples: int = 20,
) -> Iterator[dict[str, Any]]:
    """Yield clean chunks; fills ``report`` (chunks, fatal, examples) as it goes."""
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"in": 0, "dropped": 0, "by_kind": Counter(), "by_ref": Counter()}
    )
    fatal: list[dict[str, str]] = []
    shown: list[dict[str, Any]] = []
    report.update({"chunks": {}, "fatal": fatal, "examples": shown})
    for c in chunks:
        st = stats[c["source"]]
        st["in"] += 1
        hits = ref.check_text(c["text"])
        if not hits:
            yield c
            continue
        st["dropped"] += 1
        for h in hits:
            st["by_kind"][h["kind"]] += 1
            st["by_ref"][h["ref"]] += 1
            if _fatal(h, c["source"] in generated_sources):
                fatal.append({"chunk": c["id"], **h})
        if len(shown) < examples:
            shown.append({"chunk": c["id"], "hits": hits[:5]})
    report["chunks"] = {
        s: {**v, "by_kind": dict(v["by_kind"]), "by_ref": dict(v["by_ref"])}
        for s, v in sorted(stats.items())
    }


def decontaminate_chunks(
    chunks: Iterable[dict[str, Any]],
    ref: ReferenceSet,
    *,
    generated_sources: set[str] = frozenset(),  # type: ignore[assignment]
    examples: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {}
    kept = list(
        iter_clean_chunks(
            chunks, ref, report, generated_sources=generated_sources, examples=examples
        )
    )
    return kept, report


def decontaminate_probes(
    probes: Iterable[dict[str, Any]],
    ref: ReferenceSet,
    *,
    generated_sources: set[str] = frozenset(),  # type: ignore[assignment]
    jaccard_threshold: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    fatal, shown = [], []
    for p in probes:
        st = stats[p["source"]]
        st["in"] += 1
        answers = " ".join(f["value"] for f in p.get("facts", []))
        hits = ref.check_text(f"{p['question']}\n{answers}", ngrams=False)
        dup = ref.near_duplicate_question(p["question"], jaccard_threshold)
        if dup:
            hits.append({"kind": "eval_question", "ref": dup, "match": p["question"]})
        if not hits:
            kept.append(p)
            continue
        st["dropped"] += 1
        for h in hits:
            st[h["kind"]] += 1
            if _fatal(h, p["source"] in generated_sources):
                fatal.append({"probe": p["id"], **h})
        if len(shown) < 20:
            shown.append({"probe": p["id"], "hits": hits[:5]})
    return kept, {
        "probes": {s: dict(v) for s, v in sorted(stats.items())},
        "fatal": fatal,
        "examples": shown,
    }
