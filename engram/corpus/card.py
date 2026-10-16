"""Stage 9: data-card statistics (``card.json``, ``DATA_CARD.md``).

Counts are measured with the build's tokenizer. With the default whitespace tokenizer they are
**approximate** (words, not Qwen3 tokens), and the card says so.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from engram.corpus import sources as src
from engram.corpus.io import read_jsonl
from engram.corpus.tokenize import Tokenizer


def _registry_row(key: str) -> dict[str, Any]:
    s = src.REGISTRY.get(key)
    if s is None:
        return {"licence": None, "hf_id": None, "slice": None, "verdict": None}
    return {"licence": s.licence, "hf_id": s.hf_id, "slice": s.slice, "verdict": s.verdict}


def doc_stats(docs: Iterable[dict[str, Any]], tok: Tokenizer) -> dict[str, Any]:
    """Per-source document counts, token counts and licences for raw or filtered docs."""
    per: dict[str, dict[str, Any]] = {}
    for d in docs:
        s = per.setdefault(
            d["source"],
            {**_registry_row(d["source"]), "docs": 0, "tokens": 0, "doc_licences": Counter()},
        )
        n = tok.count(d["text"])
        s["docs"] += 1
        s["tokens"] += n
        s["doc_licences"][d.get("licence") or "?"] += 1
    for s in per.values():
        s["mean_doc_tokens"] = round(s["tokens"] / s["docs"], 1) if s["docs"] else 0
        s["doc_licences"] = dict(s["doc_licences"].most_common(10))
    return {
        "tokenizer": tok.name,
        "approximate": tok.approximate,
        "sources": dict(sorted(per.items())),
        "total_docs": sum(s["docs"] for s in per.values()),
        "total_tokens": sum(s["tokens"] for s in per.values()),
    }


def chunk_stats(chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"chunks": 0, "tokens": 0, "docs": set(), "splits": Counter(), "sizes": Counter()}
    )
    for c in chunks:
        s = per[c["source"]]
        s["chunks"] += 1
        s["tokens"] += c["n_tokens"]
        s["docs"].add(c["doc_id"])
        s["splits"][c["split"]] += c["n_tokens"]
        s["sizes"][str(c["chunk_size"])] += 1
    return {
        k: {
            "chunks": v["chunks"],
            "tokens": v["tokens"],
            "docs": len(v["docs"]),
            "tokens_by_split": dict(v["splits"]),
            "chunk_sizes": dict(v["sizes"]),
        }
        for k, v in sorted(per.items())
    }


def episode_stats(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    n = 0
    by_split: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    mixed = 0
    ks: list[int] = []
    ptype: Counter[str] = Counter()
    plag: Counter[str] = Counter()
    write_tokens = target_tokens = 0
    for e in episodes:
        n += 1
        by_split[e["split"]] += 1
        sizes[str(e["chunk_size"])] += 1
        mixed += bool(e["mixed"])
        ks.append(e["k"])
        write_tokens += sum(w["n_tokens"] for w in e["writes"])
        target_tokens += e["target"]["n_tokens"]
        for p in e["probes"]:
            ptype[p["type"]] += 1
            plag[p["lag_bucket"]] += 1
    return {
        "episodes": n,
        "by_split": dict(by_split),
        "chunk_sizes": dict(sizes),
        "mixed": mixed,
        "mean_k": round(sum(ks) / n, 2) if n else 0,
        "max_k": max(ks, default=0),
        "write_tokens": write_tokens,
        "target_tokens": target_tokens,
        "probes": sum(ptype.values()),
        "probes_by_type": dict(ptype),
        "probes_by_lag": dict(plag),
    }


def build_card(
    out: Path, tok: Tokenizer, *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Assemble the card from a build directory (whatever stages exist)."""
    card: dict[str, Any] = {"tokenizer": tok.name, "approximate": tok.approximate}
    doc_files = sorted((out / "docs").glob("*.jsonl"))
    if doc_files:
        card["docs"] = doc_stats((d for f in doc_files for d in read_jsonl(f)), tok)
    if (out / "chunks.jsonl").exists():
        card["chunks_before_decontam"] = chunk_stats(read_jsonl(out / "chunks.jsonl"))
    kept = out / "decontam" / "chunks.jsonl"
    if kept.exists():
        card["chunks"] = chunk_stats(read_jsonl(kept))
    if (out / "episodes.jsonl").exists():
        card["episodes"] = episode_stats(read_jsonl(out / "episodes.jsonl"))
    if config:
        card["mix"] = mix_table(config, card.get("chunks", {}))
    return card


def mix_table(config: dict[str, Any], chunks: dict[str, Any]) -> list[dict[str, Any]]:
    target = float(config.get("target_tokens", 0))
    rows: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for entry in config.get("sources", []):
        key = entry["key"]  # chunks are keyed by source, so pg19 val + test share one row
        planned = entry.get("tokens")
        if planned is None and entry.get("share") is not None:
            planned = float(entry["share"]) * target
        if key in by_key:
            row = by_key[key]
            if planned is not None:
                row["planned_tokens"] = (row["planned_tokens"] or 0) + planned
            continue
        by_key[key] = row = {
            "source": key,
            "slice": entry.get("slice") or _registry_row(key)["slice"],
            "planned_tokens": planned,
            "built_tokens": chunks.get(key, {}).get("tokens", 0),
            "use": entry.get("use", True),
        }
        rows.append(row)
    return rows


def render_markdown(card: dict[str, Any], manifest: dict[str, Any] | None = None) -> str:
    unit = "approximate tokens, not Qwen3" if card.get("approximate") else "measured tokens"
    lines = ["# Engram Phase 1 corpus: data card", ""]
    if manifest:
        lines += [
            f"- Build: `{manifest.get('name')}`, seed {manifest.get('seed')}, "
            f"git `{manifest.get('git_commit')}`, built {manifest.get('built_at')}"
        ]
    lines += [f"- Tokenizer: `{card.get('tokenizer')}` ({unit})", ""]
    chunks = card.get("chunks", {})
    docs = card.get("docs", {}).get("sources", {})
    if docs or chunks:
        lines += [
            "## Sources",
            "",
            "| Source | Slice | Licence | Docs | Chunks | Tokens (kept) |",
            "|---|---|---|---|---|---|",
        ]
        for key in sorted(set(docs) | set(chunks)):
            reg = _registry_row(key)
            d = docs.get(key, {})
            c = chunks.get(key, {})
            lic = (reg["licence"] or ", ".join(d.get("doc_licences", {}))).replace("|", "/")
            lines.append(
                f"| `{key}` | {reg['slice'] or ''} | {lic} | {d.get('docs', 0)} | "
                f"{c.get('chunks', 0)} | {c.get('tokens', 0):,} |"
            )
        lines.append("")
    if card.get("mix"):
        lines += [
            "## Mix (planned vs built)",
            "",
            "| Source | Slice | Planned | Built |",
            "|---|---|---|---|",
        ]
        for r in card["mix"]:
            planned = f"{r['planned_tokens']:,.0f}" if r["planned_tokens"] is not None else "-"
            lines.append(
                f"| `{r['source']}` | {r['slice'] or ''} | {planned} | {r['built_tokens']:,} |"
            )
        lines.append("")
    if card.get("episodes"):
        e = card["episodes"]
        lines += [
            "## Episodes",
            "",
            f"- {e['episodes']} episodes ({e['mixed']} mixed), by split {e['by_split']}",
            f"- chunk sizes {e['chunk_sizes']}; mean k {e['mean_k']}, max k {e['max_k']}",
            f"- {e['probes']} probes; by type {e['probes_by_type']}; by lag {e['probes_by_lag']}",
            "",
        ]
    contam = card.get("contamination")
    if contam:
        lines += ["## Decontamination", ""]
        for s, v in contam.get("chunks", {}).items():
            lines.append(f"- `{s}`: dropped {v['dropped']} of {v['in']} chunks {v['by_kind']}")
        for s, v in contam.get("probes", {}).items():
            lines.append(f"- `{s}` probes: {v}")
        lines.append(f"- fatal hits: {len(contam.get('fatal', []))}")
        lines.append("")
    return "\n".join(lines)
