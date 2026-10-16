"""Stage 5b: write-then-recall episodes (research/phase1-corpus.md §b).

An episode is a contiguous run of k write chunks from one document (k uniform in 1..K, with
K x chunk size <= ``max_context``), then a target chunk (k+1) for next-token loss, then probes
about the writes. Documents longer than one episode give several episodes, each with a fresh
memory. A gap in chunk indices (a chunk dropped by decontamination) ends a run, so no episode
spans a dropped chunk.

A fraction ``p_mixed`` of episodes are *mixed*: 2-4 single-document episodes of the same split
and chunk size concatenated, with ``boundary_before`` marking each new document, and probes about
each document (interference within one memory).

Probes are document-level records (``facts`` with character spans, see bios.py and
llm_probes.py). For each episode the gold is the *latest* value among the facts the writes
contain; earlier values seen in the writes become ``reject`` (stale-value-wrong). Abstention
probes need their ``anchor`` inside the writes. Lag = k-1-j for evidence in write j (0-based),
bucketed {0, 1-3, 4-15, 16+}; at most ``max_probes`` per episode, spread across buckets.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from engram.corpus.io import sha256

LAG_BUCKETS = ((0, 0, "0"), (1, 3, "1-3"), (4, 15, "4-15"), (16, 10**9, "16+"))
_WRITE_FIELDS = ("id", "source", "doc_id", "licence", "text", "n_tokens", "char_offsets", "sha256")


def lag_bucket(lag: int) -> str:
    for lo, hi, name in LAG_BUCKETS:
        if lo <= lag <= hi:
            return name
    raise ValueError(lag)


def runs(chunks: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Contiguous chunk runs per document, in document order."""
    by_doc: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_doc[(c["source"], c["doc_id"], str(c["chunk_size"]))].append(c)
    out = []
    for key in sorted(by_doc):
        cs = sorted(by_doc[key], key=lambda c: c["chunk_index"])
        cur = [cs[0]]
        for c in cs[1:]:
            if c["chunk_index"] == cur[-1]["chunk_index"] + 1:
                cur.append(c)
            else:
                out.append(cur)
                cur = [c]
        out.append(cur)
    return out


def _k_max(chunk_size: int | str, k_max: int, max_context: int, session_max: int) -> int:
    size = session_max if chunk_size == "session" else int(chunk_size)
    return max(1, min(k_max, max_context // size))


def _write(c: dict[str, Any], boundary: bool = False) -> dict[str, Any]:
    w = {("chunk_id" if k == "id" else k): c[k] for k in _WRITE_FIELDS if k in c}
    w["boundary_before"] = boundary
    return w


def attach_probes(
    writes: list[dict[str, Any]], probes_by_doc: dict[tuple[str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    k = len(writes)
    out = []
    docs = []
    for w in writes:
        key = (w["source"], w["doc_id"])
        if key not in docs:
            docs.append(key)

    def locate(key: tuple[str, str], start: int, end: int) -> int | None:
        for j, w in enumerate(writes):
            cs, ce = w["char_offsets"]
            if (w["source"], w["doc_id"]) == key and cs <= start and end <= ce:
                return j
        return None

    for key in docs:
        for p in probes_by_doc.get(key, ()):
            base = {
                "id": p["id"],
                "question": p["question"],
                "source_type": p["type"],
                "attribute": p.get("attribute"),
                "generator": p.get("generator"),
            }
            if p.get("facts"):
                inside = []
                for f in sorted(p["facts"], key=lambda f: f["start"]):
                    j = locate(key, f["start"], f["end"])
                    if j is not None:
                        inside.append((f, j))
                if not inside:
                    continue
                last, j = inside[-1]
                gold = last["value"]
                reject = list(
                    dict.fromkeys(f["value"] for f, _ in inside[:-1] if f["value"] != gold)
                )
                kind = p["type"]
                if kind == "update" and not reject:
                    kind = "fact"
                cs = writes[j]["char_offsets"][0]
                evidence = writes[j]["text"][last["start"] - cs : last["end"] - cs]
                accept = list(dict.fromkeys([gold, *p.get("accept", [])]))
                out.append(
                    {
                        **base,
                        "type": kind,
                        "answer": gold,
                        "accept": accept,
                        "reject": reject,
                        "write_index": j,
                        "evidence": evidence,
                    }
                )
            elif p.get("anchor"):
                j = locate(key, *p["anchor"])
                if j is None:
                    continue
                out.append(
                    {
                        **base,
                        "type": "abstain",
                        "answer": None,
                        "accept": [],
                        "reject": [],
                        "write_index": j,
                        "evidence": None,
                    }
                )
    for p in out:
        p["lag"] = k - 1 - p["write_index"]
        p["lag_bucket"] = lag_bucket(p["lag"])
    return out


def select_probes(probes: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Up to n probes, round-robin across lag buckets so recall is measured against distance."""
    if len(probes) <= n:
        return probes
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in probes:
        buckets[p["lag_bucket"]].append(p)
    for b in buckets.values():
        rng.shuffle(b)
    order = [name for _, _, name in LAG_BUCKETS if name in buckets]
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n:
        for name in order:
            if buckets[name] and len(chosen) < n:
                chosen.append(buckets[name].pop())
    return sorted(chosen, key=lambda p: (p["write_index"], p["id"]))


def build_episodes(
    chunks: Iterable[dict[str, Any]],
    probes: Iterable[dict[str, Any]] = (),
    *,
    seed: int = 0,
    k_max: int = 128,
    max_context: int = 131072,
    session_max: int = 4096,
    p_mixed: float = 0.2,
    mixed_docs: tuple[int, int] = (2, 4),
    max_probes: int = 16,
) -> list[dict[str, Any]]:
    probes_by_doc: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in probes:
        probes_by_doc[(p["source"], p["doc_id"])].append(p)

    raw: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for run in runs(chunks):
        rng = random.Random(
            f"episodes:{seed}:{run[0]['source']}:{run[0]['doc_id']}:{run[0]['chunk_index']}"
        )
        big_k = _k_max(run[0]["chunk_size"], k_max, max_context, session_max)
        i = 0
        while len(run) - i >= 2:
            k = min(rng.randint(1, big_k), len(run) - i - 1)
            raw.append((run[i : i + k], run[i + k]))
            i += k + 1

    rng = random.Random(f"episodes:{seed}:mix")
    rng.shuffle(raw)
    groups: list[list[tuple[list[dict[str, Any]], dict[str, Any]]]] = []
    used = [False] * len(raw)
    for a in range(len(raw)):
        if used[a]:
            continue
        used[a] = True
        group = [raw[a]]
        if p_mixed > 0 and rng.random() < p_mixed:
            want = rng.randint(*mixed_docs)
            first = raw[a][1]
            big_k = _k_max(first["chunk_size"], k_max, max_context, session_max)
            total = len(raw[a][0])
            seen_docs = {(first["source"], first["doc_id"])}
            for b in range(a + 1, len(raw)):
                if len(group) >= want:
                    break
                ws, t = raw[b]
                key = (t["source"], t["doc_id"])
                if (
                    used[b]
                    or key in seen_docs
                    or t["split"] != first["split"]
                    or t["chunk_size"] != first["chunk_size"]
                    or total + len(ws) > big_k
                ):
                    continue
                used[b] = True
                group.append(raw[b])
                seen_docs.add(key)
                total += len(ws)
        groups.append(group)

    episodes = []
    for group in groups:
        writes = []
        for g, (ws, _) in enumerate(group):
            writes += [_write(c, boundary=(g > 0 and n == 0)) for n, c in enumerate(ws)]
        target = group[-1][1]
        ids = [w["chunk_id"] for w in writes] + [target["id"]]
        eid = "ep-" + sha256("|".join(ids))[:16]
        prng = random.Random(f"probes:{seed}:{eid}")
        ps = select_probes(attach_probes(writes, probes_by_doc), max_probes, prng)
        episodes.append(
            {
                "episode_id": eid,
                "split": target["split"],
                "chunk_size": target["chunk_size"],
                "k": len(writes),
                "mixed": len(group) > 1,
                "sources": sorted({w["source"] for w in writes}),
                "doc_ids": list(dict.fromkeys(w["doc_id"] for w in writes)),
                "slices": sorted({t.get("slice") or "" for _, t in group}),
                "writes": writes,
                "target": _write(target),
                "probes": ps,
            }
        )
    return episodes


def pack(
    episodes: list[dict[str, Any]], *, seed: int = 0, shard_size: int = 1000
) -> dict[str, list[list[dict[str, Any]]]]:
    """Shuffle episodes by seed within each split and cut them into shards."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in episodes:
        by_split[e["split"]].append(e)
    out = {}
    for split in sorted(by_split):
        eps = sorted(by_split[split], key=lambda e: e["episode_id"])
        random.Random(f"pack:{seed}:{split}").shuffle(eps)
        out[split] = [eps[i : i + shard_size] for i in range(0, len(eps), shard_size)]
    return out
