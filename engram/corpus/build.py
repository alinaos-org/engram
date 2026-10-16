"""``engram-corpus build``: run the stages from a config (research/phase1-corpus.md §e).

Output layout under ``--out``::

    raw/<source>.jsonl            fetch (capped, streaming) + fetch_manifest.json
    docs/<source>.jsonl           filter (+ filter_report.json); bios writes docs/engram_bios.jsonl
    probes/bios.jsonl             exact bio probes (document-level spans)
    chunks.jsonl                  chunk
    decontam/chunks.jsonl         decontam (+ decontam/probes.jsonl, contamination_report.json)
    probes/llm_requests.jsonl     probes --dry-run (or probes/llm.jsonl with a model)
    episodes.jsonl                episodes
    shards/<split>-NNNNN.jsonl    pack
    card.json DATA_CARD.md manifest.json

Stage order differs slightly from the plan's outline: the 13-gram filter runs on chunks (after
``chunk``), so a contaminated chunk ends its episode there, as §d.4 asks. ``generate`` and
``probe``/``verify`` are accepted as aliases of ``bios`` and ``probes``.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram import __version__
from engram.corpus import bios, card, chunk, decontam, episodes, fetch, filter, llm_probes
from engram.corpus.io import read_jsonl, read_many, write_json, write_jsonl
from engram.corpus.tokenize import Tokenizer, get_tokenizer

STAGES = ["fetch", "filter", "bios", "chunk", "decontam", "probes", "episodes", "pack", "card"]
ALIASES = {"generate": "bios", "probe": "probes", "verify": "probes"}


class BuildError(RuntimeError):
    pass


@dataclass
class Context:
    config: dict[str, Any]
    out: Path
    seed: int = 0
    cap: int | None = None
    target_tokens: float | None = None
    dry_run: bool = False
    resume: bool = False
    allow_ablation: bool = False
    tokenizer: Tokenizer = field(default_factory=lambda: get_tokenizer("whitespace"))
    timings: dict[str, float] = field(default_factory=dict)
    _ref: decontam.ReferenceSet | None = None

    def specs(self) -> list[dict[str, Any]]:
        return [fetch.spec_for(e) for e in self.config.get("sources", []) if e.get("use", True)]

    def spec_by_name(self) -> dict[str, dict[str, Any]]:
        return {s.get("out_name") or s["key"]: s for s in self.specs()}

    def generated(self) -> set[str]:
        return {s["key"] for s in self.specs() if s.get("generated")}

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.config.get(name) or {})

    def reference(self) -> decontam.ReferenceSet:
        if self._ref is None:
            self._ref = make_reference(self.section("decontam"))
        return self._ref


def make_reference(cfg: dict[str, Any]) -> decontam.ReferenceSet:
    external = cfg.get("external", decontam.DEFAULT_EXTERNAL)
    return decontam.build_reference(
        n=int(cfg.get("n", 13)),
        bench=cfg.get("bench", {}),  # {} = BenchConfig defaults; null = skip engram-bench
        bench_dirs=cfg.get("bench_dirs", []),
        external=external,
    )


# -- stages -------------------------------------------------------------------------------------


def stage_fetch(ctx: Context) -> dict[str, Any]:
    specs = [s for s in ctx.specs() if not s.get("generated")]
    target = ctx.target_tokens if ctx.cap is None else None
    man = fetch.fetch(
        specs,
        ctx.out,
        cap=ctx.cap,
        tokenizer=ctx.tokenizer,
        target_tokens=target,
        allow_ablation=ctx.allow_ablation,
    )
    write_json(ctx.out / "fetch_manifest.json", man)
    return {"docs": {Path(e["file"]).stem: e["n_docs"] for e in man["sources"]}}


def stage_filter(ctx: Context) -> dict[str, Any]:
    specs = {n: s for n, s in ctx.spec_by_name().items() if not s.get("generated")}
    raw = {n: ctx.out / "raw" / f"{n}.jsonl" for n in specs}
    cfg = ctx.section("filter")
    cfg.setdefault("min_chunk_tokens", min(ctx.section("chunk").get("sizes", [1024])))
    held = filter.holdout_ids(
        d for n, s in specs.items() if s["key"] == "pg19" for d in read_jsonl(raw[n])
    )
    f = filter.DocFilter(cfg, ctx.tokenizer, held)
    # hold-out books first, so a Gutenberg copy of a PG-19 val/test book is the duplicate
    order = sorted(specs, key=lambda n: specs[n]["key"] != "pg19")
    kept = {}
    for name in order:
        kept[name] = write_jsonl(
            ctx.out / "docs" / f"{name}.jsonl", f.apply(name, specs[name], read_jsonl(raw[name]))
        )
    write_json(ctx.out / "filter_report.json", f.report())
    return {"kept": kept}


def stage_bios(ctx: Context) -> dict[str, Any]:
    spec = next((s for s in ctx.specs() if s["key"] == bios.SOURCE), None)
    if spec is None:
        return {"skipped": "no engram_bios source in config"}
    cfg = ctx.section("bios")
    people = int(cfg.get("people_per_doc", 40))
    kw = {
        "people": people,
        "p_update": float(cfg.get("p_update", 0.3)),
        "p_extra": float(cfg.get("p_extra", 0.5)),
    }
    n_docs = cfg.get("docs")
    if n_docs is None:
        tokens = spec.get("tokens")
        if tokens is None and spec.get("share") is not None:
            tokens = float(spec["share"]) * float(ctx.target_tokens or 0)
        sample, _ = bios.generate_doc(ctx.seed, 0, **kw)
        n_docs = max(1, math.ceil(float(tokens or 0) / max(1, ctx.tokenizer.count(sample["text"]))))
    if ctx.cap is not None:
        n_docs = min(int(n_docs), ctx.cap)
    docs, probes = bios.generate(ctx.seed, int(n_docs), **kw)
    problems = bios.check_disjoint(docs, probes)
    if problems:
        raise BuildError("bios overlap engram-bench: " + "; ".join(problems[:5]))
    write_jsonl(ctx.out / "docs" / f"{bios.SOURCE}.jsonl", docs)
    write_jsonl(ctx.out / "probes" / "bios.jsonl", probes)
    return {"docs": len(docs), "probes": len(probes)}


def stage_chunk(ctx: Context) -> dict[str, Any]:
    cfg = ctx.section("chunk")
    specs = ctx.specs()
    files = sorted((ctx.out / "docs").glob("*.jsonl"))
    n = write_jsonl(
        ctx.out / "chunks.jsonl",
        chunk.chunk_docs(
            read_many(files),
            ctx.tokenizer,
            sizes=[int(s) for s in cfg.get("sizes", [1024, 2048, 4096])],
            seed=ctx.seed,
            tolerance=float(cfg.get("tolerance", 0.1)),
            modes={s["key"]: s.get("chunk_mode", "tokens") for s in specs},
            slices={s["key"]: s.get("slice") for s in specs},
            session_max=int(cfg.get("session_max", 4096)),
            val=float(cfg.get("val", 0.01)),
            test=float(cfg.get("test", 0.01)),
        ),
    )
    return {"chunks": n}


def stage_decontam(ctx: Context) -> dict[str, Any]:
    cfg = ctx.section("decontam")
    ref = ctx.reference()
    gen = ctx.generated()
    background_files = [f for f in sorted((ctx.out / "docs").glob("*.jsonl")) if f.stem not in gen]
    stock = ref.drop_stock_phrases(
        (d["text"] for d in read_many(background_files)),
        max_df=float(cfg.get("background_max_df", 0.001)),
        min_docs=int(cfg.get("background_min_docs", 1000)),
    )
    report: dict[str, Any] = {}
    n_kept = write_jsonl(
        ctx.out / "decontam" / "chunks.jsonl",
        decontam.iter_clean_chunks(
            read_jsonl(ctx.out / "chunks.jsonl"), ref, report, generated_sources=gen
        ),
    )
    probe_files = [ctx.out / "probes" / "bios.jsonl"]
    probes = list(read_many(p for p in probe_files if p.exists()))
    kept_p, preport = decontam.decontaminate_probes(
        probes,
        ref,
        generated_sources=gen,
        jaccard_threshold=float(cfg.get("probe_jaccard", 0.8)),
    )
    write_jsonl(ctx.out / "decontam" / "probes.jsonl", kept_p)
    full = {
        "reference": ref.summary(),
        "stock_phrases": stock,
        **report,
        "probes": preport["probes"],
        "fatal": report["fatal"] + preport["fatal"],
        "probe_examples": preport["examples"],
    }
    write_json(ctx.out / "contamination_report.json", full)
    if full["fatal"]:
        raise BuildError(
            f"decontamination found {len(full['fatal'])} fatal hit(s) (canary, user name, or a "
            f"template in generated data); see {ctx.out / 'contamination_report.json'}"
        )
    return {"chunks_kept": n_kept, "probes_kept": len(kept_p)}


def _chat(spec: dict[str, Any] | None, cache: Path) -> Any:
    """A cached OpenAI-compatible client for a {model, revision, base_url} spec, or None."""
    if not spec or not spec.get("base_url"):
        return None
    from engram.models.cache import CachedModel
    from engram.models.chat import OpenAIChat

    model_id = spec["model"] + (f"@{spec['revision']}" if spec.get("revision") else "")
    inner = OpenAIChat(
        spec["model"], base_url=spec["base_url"], api_key_env=spec.get("api_key_env")
    )
    return CachedModel(_Renamed(inner, model_id), path=cache)


def make_generator(cfg: dict[str, Any], cache: Path) -> llm_probes.LLMProbeGenerator | None:
    model = _chat(cfg.get("generator"), cache)
    if model is None:
        return None
    n_min, n_max = cfg.get("items_per_chunk", [3, 5])
    return llm_probes.LLMProbeGenerator(model, n_min, n_max)


def make_verifier(cfg: dict[str, Any], cache: Path) -> llm_probes.ChatVerifier | None:
    """Open-book verifier plus closed-book floor; both must be configured to verify."""
    open_model = _chat(cfg.get("verifier"), cache)
    closed_model = _chat(cfg.get("closed_book"), cache)
    if open_model is None or closed_model is None:
        return None
    return llm_probes.ChatVerifier(open_model, closed_model)


def _live(generator: llm_probes.LLMProbeGenerator) -> bool:
    return not isinstance(generator.model, llm_probes.UnrunModel)


@dataclass
class _Renamed:
    """Cache LLM completions by (model, revision, prompt): the revision is part of the name."""

    inner: Any
    name: str

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 512) -> str:
        return self.inner.complete(messages, max_tokens=max_tokens)


def stage_probes(ctx: Context) -> dict[str, Any]:
    cfg = ctx.section("probes")
    llm_sources = {s["key"] for s in ctx.specs() if s.get("probes", "llm") == "llm"}
    splits = set(cfg.get("splits", ["train", "val", "test"]))
    chunks = [
        c
        for c in read_jsonl(ctx.out / "decontam" / "chunks.jsonl")
        if c["source"] in llm_sources and c["split"] in splits
    ]
    if cfg.get("max_chunks") is not None:
        chunks = chunks[: int(cfg["max_chunks"])]
    generator = None if ctx.dry_run else make_generator(cfg, ctx.out / "llm_cache.sqlite")
    if generator is None:  # dry run: same prompts, the configured model's name, no calls
        name = (cfg.get("generator") or {}).get("model", "unset")
        n_min, n_max = cfg.get("items_per_chunk", [3, 5])
        generator = llm_probes.LLMProbeGenerator(llm_probes.UnrunModel(name), n_min, n_max)
    live = not ctx.dry_run and _live(generator)
    verifier = make_verifier(cfg, ctx.out / "llm_cache.sqlite") if live else None
    summary = llm_probes.run_stage(
        chunks, ctx.out / "probes", generator=generator, verifier=verifier, dry_run=not live
    )
    summary["verified"] = verifier is not None
    if not summary["dry_run"]:
        ref = ctx.reference()
        probes = list(read_jsonl(ctx.out / "probes" / "llm.jsonl"))
        kept, rep = decontam.decontaminate_probes(
            probes,
            ref,
            generated_sources=ctx.generated(),
            jaccard_threshold=float(ctx.section("decontam").get("probe_jaccard", 0.8)),
        )
        write_jsonl(ctx.out / "probes" / "llm.jsonl", kept)
        summary["probe_decontam"] = rep["probes"]
        rpath = ctx.out / "contamination_report.json"
        if rpath.exists():
            full = json.loads(rpath.read_text(encoding="utf-8"))
            full["llm_probes"] = rep["probes"]
            full["fatal"] = full.get("fatal", []) + rep["fatal"]
            write_json(rpath, full)
        if rep["fatal"]:
            raise BuildError(f"LLM probes hit {len(rep['fatal'])} canary/name/template(s)")
    return summary


def stage_episodes(ctx: Context) -> dict[str, Any]:
    cfg = ctx.section("episodes")
    chunk_cfg = ctx.section("chunk")
    probe_files = [ctx.out / "decontam" / "probes.jsonl", ctx.out / "probes" / "llm.jsonl"]
    eps = episodes.build_episodes(
        read_jsonl(ctx.out / "decontam" / "chunks.jsonl"),
        read_many(p for p in probe_files if p.exists()),
        seed=ctx.seed,
        k_max=int(cfg.get("k_max", 128)),
        max_context=int(cfg.get("max_context", 131072)),
        session_max=int(chunk_cfg.get("session_max", 4096)),
        p_mixed=float(cfg.get("p_mixed", 0.2)),
        mixed_docs=tuple(cfg.get("mixed_docs", (2, 4))),  # type: ignore[arg-type]
        max_probes=int(cfg.get("max_probes", 16)),
    )
    write_jsonl(ctx.out / "episodes.jsonl", eps)
    return {"episodes": len(eps), "probes": sum(len(e["probes"]) for e in eps)}


def stage_pack(ctx: Context) -> dict[str, Any]:
    cfg = ctx.section("pack")
    shards = episodes.pack(
        list(read_jsonl(ctx.out / "episodes.jsonl")),
        seed=ctx.seed,
        shard_size=int(cfg.get("shard_size", 1000)),
    )
    out = {}
    for split, parts in shards.items():
        for i, part in enumerate(parts):
            write_jsonl(ctx.out / "shards" / f"{split}-{i:05d}.jsonl", part)
        out[split] = len(parts)
    return {"shards": out}


def _git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def stage_card(ctx: Context) -> dict[str, Any]:
    c = card.build_card(ctx.out, ctx.tokenizer, config=ctx.config)
    rpath = ctx.out / "contamination_report.json"
    if rpath.exists():
        rep = json.loads(rpath.read_text(encoding="utf-8"))
        c["contamination"] = {k: rep.get(k) for k in ("chunks", "probes", "fatal")}
    fman = ctx.out / "fetch_manifest.json"
    probes_cfg = ctx.section("probes")
    manifest = {
        "name": ctx.config.get("name"),
        "seed": ctx.seed,
        "engram_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": ctx.tokenizer.name,
        "cap": ctx.cap,
        "target_tokens": ctx.target_tokens,
        "dry_run": ctx.dry_run,
        "models": {k: probes_cfg.get(k) for k in ("generator", "verifier", "closed_book")},
        "bench_session_writer": probes_cfg.get("bench_session_writer"),
        "fetch": json.loads(fman.read_text(encoding="utf-8")) if fman.exists() else None,
        "stage_seconds": _timings(ctx),
        "config": ctx.config,
    }
    write_json(ctx.out / "card.json", c)
    write_json(ctx.out / "manifest.json", manifest)
    (ctx.out / "DATA_CARD.md").write_text(card.render_markdown(c, manifest) + "\n", "utf-8")
    return {"card": str(ctx.out / "DATA_CARD.md")}


def _timings(ctx: Context) -> dict[str, float]:
    out = {}
    for f in sorted((ctx.out / ".stages").glob("*.json")):
        out[f.stem] = json.loads(f.read_text(encoding="utf-8")).get("seconds")
    return out


STAGE_FUNCS = {
    "fetch": stage_fetch,
    "filter": stage_filter,
    "bios": stage_bios,
    "chunk": stage_chunk,
    "decontam": stage_decontam,
    "probes": stage_probes,
    "episodes": stage_episodes,
    "pack": stage_pack,
    "card": stage_card,
}


def parse_stages(spec: str | None) -> list[str]:
    if not spec:
        return list(STAGES)
    wanted = []
    for s in spec.split(","):
        s = ALIASES.get(s.strip(), s.strip())
        if s not in STAGES:
            raise ValueError(f"unknown stage {s!r}; stages: {', '.join(STAGES)}")
        if s not in wanted:
            wanted.append(s)
    return [s for s in STAGES if s in wanted]


def run(ctx: Context, stages: list[str], log: Any = print) -> dict[str, Any]:
    ctx.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in stages:
        marker = ctx.out / ".stages" / f"{name}.json"
        if ctx.resume and marker.exists() and name != "card":
            log(f"[{name}] done already (resume)")
            continue
        t0 = time.time()
        result = STAGE_FUNCS[name](ctx)
        seconds = round(time.time() - t0, 3)
        write_json(marker, {"seconds": seconds, "result": result})
        log(f"[{name}] {json.dumps(result)} ({seconds}s)")
        results[name] = result
    return results
