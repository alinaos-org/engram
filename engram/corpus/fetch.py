"""Stage 1: fetch documents, capped, streaming.

Three kinds of source spec (a config entry, merged over its registry row):

* ``hf``: Hugging Face ``datasets`` in streaming mode at a pinned revision. Only the first
  ``cap`` documents (or as many as fit the token budget) are ever read.
* ``http_jsonl``: a JSONL or JSONL.gz URL, streamed line by line with ``urllib``.
* ``local``: a JSONL file of ready-made documents (tests, fixtures, pre-fetched shards).

Every document becomes ``{source, doc_id, text, licence, meta}``. Lineage-excluded datasets are
refused, and ablation/validation sources need ``allow_ablation``.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from engram.corpus import sources as src
from engram.corpus.io import read_jsonl, write_jsonl
from engram.corpus.tokenize import Tokenizer

_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def spec_for(entry: dict[str, Any]) -> dict[str, Any]:
    """Merge a config source entry over its registry row (if any)."""
    key = entry["key"]
    base = asdict(src.REGISTRY[key]) if key in src.REGISTRY else {"key": key}
    merged = {**base, **entry}
    merged.setdefault("kind", "local" if merged.get("path") else "hf")
    return merged


def check_allowed(spec: dict[str, Any], *, allow_ablation: bool = False) -> None:
    hf_id = spec.get("hf_id")
    if hf_id in src.LINEAGE_EXCLUSIONS:
        raise ValueError(f"{hf_id} is lineage-excluded: {src.LINEAGE_EXCLUSIONS[hf_id]}")
    verdict = spec.get("verdict", "use")
    if verdict in ("ablation", "validation", "optional") and not allow_ablation:
        raise ValueError(
            f"source {spec['key']} has verdict {verdict!r}; pass allow_ablation to fetch it"
        )


def _get(d: dict[str, Any], dotted: str | None) -> Any:
    if not dotted:
        return None
    if "|" in dotted:  # alternatives: the first one present wins
        for alt in dotted.split("|"):
            v = _get(d, alt)
            if v:
                return v
        return None
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, str):
            try:
                cur = json.loads(cur)
            except ValueError:
                return None
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _conversation_chronicles(row: dict[str, Any]) -> str:
    """Render one ConversationChronicles episode: one paragraph per session.

    The session's time gap ("A few weeks after") is kept as a header line (§b).
    """
    gaps = row.get("time_interval") or []
    if isinstance(gaps, str):
        gaps = [gaps]
    paras = []
    for i, ordinal in enumerate(_ORDINALS):
        dialogue = row.get(f"{ordinal}_session_dialogue")
        if not dialogue:
            continue
        speakers = row.get(f"{ordinal}_session_speakers") or []
        # rows carry one interval per session, the first being "Start"
        gap = gaps[i] if i < len(gaps) else None
        header = f"[Session {i + 1}]" if i == 0 or not gap else f"[Session {i + 1}: {gap}]"
        lines = [header]
        for j, turn in enumerate(dialogue):
            who = speakers[j] if j < len(speakers) else ("A" if j % 2 == 0 else "B")
            lines.append(f"{who}: {' '.join(str(turn).split())}")
        paras.append("\n".join(lines))
    return "\n\n".join(paras)


def to_doc(row: dict[str, Any], spec: dict[str, Any], n: int) -> dict[str, Any] | None:
    fmt = spec.get("text_format", "text")
    meta: dict[str, Any] = {}
    if fmt == "conversation_chronicles":
        text = _conversation_chronicles(row)
        doc_id = str(row.get("dataID") or row.get("id") or n)
        meta["relationship"] = row.get("relationship")
    elif fmt == "pg19":
        text = row.get("text") or ""
        url = row.get("url") or ""
        m = re.search(r"/(\d+)(?:\D*)$", url)
        doc_id = m.group(1) if m else str(n)
        meta.update(
            {
                "title": row.get("short_book_title"),
                "url": url,
                "publication_date": row.get("publication_date"),
            }
        )
    elif fmt == "wikipedia":
        text = row.get("text") or ""
        doc_id = str(row.get("id") or n)
        meta.update({"title": row.get("title"), "url": row.get("url")})
        if meta["title"] and not text.startswith(str(meta["title"])):
            meta["lead"] = text[:400]
    else:
        text = row.get(spec.get("text_field") or "text") or ""
        doc_id = str(row.get(spec.get("id_field") or "id") or n)
        raw_meta = row.get("metadata")
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except ValueError:
                raw_meta = {"raw": raw_meta}
        if isinstance(raw_meta, dict):
            keep = ("license", "url", "title", "source", "language", "authors_removed")
            meta.update({k: raw_meta[k] for k in keep if k in raw_meta})
    if not text.strip():
        return None
    doc_licence = _get(row, spec.get("licence_field")) if spec.get("licence_field") else None
    return {
        "source": spec["key"],
        "doc_id": doc_id,
        "text": text,
        "licence": str(doc_licence) if doc_licence else spec.get("licence", ""),
        "meta": meta,
        **({"force_split": spec["force_split"]} if spec.get("force_split") else {}),
    }


def _hf_rows(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit("hf sources need `datasets` (uv pip install datasets)") from e
    ds = load_dataset(
        spec.get("mirror") or spec["hf_id"],
        name=spec.get("config"),
        split=spec.get("split", "train"),
        revision=spec.get("revision"),
        streaming=True,
    )
    yield from ds


def _http_jsonl_rows(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "engram-corpus"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        stream: Any = resp
        if spec["url"].endswith(".gz"):
            stream = gzip.GzipFile(fileobj=resp)
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def _local_rows(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(spec["path"])


def resolve_revision(spec: dict[str, Any]) -> str | None:
    """The commit sha the requested revision resolves to, when the Hub is reachable."""
    if spec.get("kind") != "hf":
        return spec.get("revision")
    try:
        from huggingface_hub import HfApi

        repo = spec.get("mirror") or spec["hf_id"]
        return HfApi().dataset_info(repo, revision=spec.get("revision")).sha
    except Exception:  # noqa: BLE001 - provenance is best effort; recorded as unresolved
        return None


def iter_docs(
    spec: dict[str, Any],
    *,
    cap: int | None,
    max_tokens: float | None = None,
    tokenizer: Tokenizer | None = None,
) -> Iterator[dict[str, Any]]:
    kind = spec.get("kind", "hf")
    rows: Iterable[dict[str, Any]]
    if kind == "hf":
        rows = _hf_rows(spec)
    elif kind == "http_jsonl":
        rows = _http_jsonl_rows(spec)
    elif kind == "local":
        rows = _local_rows(spec)
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    if kind == "local" and spec.get("text_format", "text") == "text":
        # local files already hold documents; keep their fields, fill in the source defaults
        def ready(r: dict[str, Any], n: int) -> dict[str, Any] | None:
            if not r.get("text", "").strip():
                return None
            return {
                "source": spec["key"],
                "doc_id": str(r.get("doc_id") or r.get("id") or n),
                "text": r["text"],
                "licence": r.get("licence") or spec.get("licence", ""),
                "meta": r.get("meta", {}),
                **({"force_split": spec["force_split"]} if spec.get("force_split") else {}),
            }

        convert = ready
    else:

        def convert(r: dict[str, Any], n: int) -> dict[str, Any] | None:
            return to_doc(r, spec, n)

    used = 0.0
    produced = 0
    for n, row in enumerate(rows):
        if cap is not None and produced >= cap:
            break
        doc = convert(row, n)
        if doc is None:
            continue
        if max_tokens is not None and tokenizer is not None:
            used += tokenizer.count(doc["text"])
            if used > max_tokens and produced > 0:
                break
        produced += 1
        yield doc


def fetch(
    specs: list[dict[str, Any]],
    out_dir: Path,
    *,
    cap: int | None,
    tokenizer: Tokenizer | None = None,
    target_tokens: float | None = None,
    allow_ablation: bool = False,
) -> dict[str, Any]:
    """Fetch every spec into ``out_dir/raw/<key>.jsonl``; returns the fetch manifest."""
    manifest: dict[str, Any] = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    entries = []
    for spec in specs:
        check_allowed(spec, allow_ablation=allow_ablation)
        if spec.get("kind") in ("hf", "http_jsonl") and cap is None and target_tokens is None:
            raise ValueError(
                f"{spec['key']}: refusing an uncapped network fetch; pass --cap or a token budget"
            )
        budget = None
        if target_tokens is not None and spec.get("share") is not None:
            budget = float(spec["share"]) * target_tokens
        elif spec.get("tokens") is not None:
            budget = float(spec["tokens"])
        name = spec.get("out_name") or spec["key"]
        path = out_dir / "raw" / f"{name}.jsonl"
        t0 = time.time()
        docs = iter_docs(spec, cap=cap, max_tokens=budget, tokenizer=tokenizer)
        n = write_jsonl(path, docs)
        entries.append(
            {
                "key": spec["key"],
                "file": str(path.relative_to(out_dir)),
                "kind": spec.get("kind"),
                "hf_id": spec.get("hf_id"),
                "fetched_from": spec.get("mirror")
                or spec.get("hf_id")
                or spec.get("url")
                or spec.get("path"),
                "config": spec.get("config"),
                "split": spec.get("split"),
                "revision_requested": spec.get("revision"),
                "revision_resolved": resolve_revision(spec) if spec.get("kind") == "hf" else None,
                "licence": spec.get("licence"),
                "card_url": spec.get("card_url"),
                "cap": cap,
                "token_budget": budget,
                "n_docs": n,
                "seconds": round(time.time() - t0, 2),
            }
        )
    manifest["sources"] = entries
    return manifest
