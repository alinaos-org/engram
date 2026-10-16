"""engram-corpus: rebuild the Phase 1 training corpus (research/phase1-corpus.md §e).

    engram-corpus build --config configs/corpus-v0.json --out data/phase1/v0 --seed 0 \\
        [--stages fetch,filter,bios,chunk,decontam,probes,episodes,pack,card] \\
        [--cap 5] [--target-tokens 10e9] [--tokenizer whitespace] [--dry-run] [--resume]

    engram-corpus fetch --config configs/corpus-v0.json --out data/phase1/smoke --cap 5
    engram-corpus bios --out data/phase1/bios --docs 10 --people 40 --seed 0
    engram-corpus chunk --docs data/phase1/bios/docs/engram_bios.jsonl --out chunks.jsonl
    engram-corpus decontam --chunks chunks.jsonl --out kept.jsonl --report report.json
    engram-corpus probes --chunks kept.jsonl --out probes/ --dry-run
    engram-corpus episodes --chunks kept.jsonl --probes probes/bios.jsonl --out episodes.jsonl
    engram-corpus card --dir data/phase1/v0       (or --docs raw/*.jsonl)

Network fetches refuse to run without ``--cap`` or a token budget. ``--dry-run`` never calls a
model: the probe stage writes its requests and a cost estimate instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engram.corpus import bios, build, card, chunk, decontam, episodes, fetch, llm_probes
from engram.corpus import sources as src
from engram.corpus.io import load_config, read_jsonl, read_many, write_json, write_jsonl
from engram.corpus.tokenize import get_tokenizer


def _docs(paths: list[str]):
    """Documents from JSONL files; a missing ``source`` defaults to the file's stem."""
    for path in paths:
        for n, d in enumerate(read_jsonl(path)):
            d.setdefault("source", Path(path).stem)
            d.setdefault("doc_id", str(d.get("id", n)))
            d.setdefault("licence", "")
            yield d


def _sizes(s: str) -> list[int]:
    return [int(float(x)) for x in s.split(",") if x.strip()]


def _cmd_build(a: argparse.Namespace) -> int:
    config = load_config(a.config)
    tok = get_tokenizer(a.tokenizer or config.get("tokenizer"))
    target = a.target_tokens if a.target_tokens is not None else config.get("target_tokens")
    ctx = build.Context(
        config=config,
        out=Path(a.out),
        seed=a.seed if a.seed is not None else int(config.get("seed", 0)),
        cap=a.cap,
        target_tokens=float(target) if target is not None else None,
        dry_run=a.dry_run,
        resume=a.resume,
        allow_ablation=a.allow_ablation,
        tokenizer=tok,
    )
    try:
        build.run(ctx, build.parse_stages(a.stages), log=lambda m: print(m, file=sys.stderr))
    except build.BuildError as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_fetch(a: argparse.Namespace) -> int:
    if a.config:
        entries = load_config(a.config).get("sources", [])
    else:
        entries = [{"key": k} for k in a.source]
    if a.source and a.config:
        entries = [e for e in entries if e["key"] in a.source or e.get("out_name") in a.source]
    specs = [fetch.spec_for(e) for e in entries if e.get("use", True)]
    specs = [s for s in specs if not s.get("generated")]
    man = fetch.fetch(
        specs,
        Path(a.out),
        cap=a.cap,
        tokenizer=get_tokenizer(a.tokenizer),
        target_tokens=a.target_tokens,
        allow_ablation=a.allow_ablation,
    )
    write_json(Path(a.out) / "fetch_manifest.json", man)
    print(json.dumps({Path(e["file"]).stem: e["n_docs"] for e in man["sources"]}))
    return 0


def _cmd_filter(a: argparse.Namespace) -> int:
    config = load_config(a.config)
    ctx = build.Context(config=config, out=Path(a.out), tokenizer=get_tokenizer(a.tokenizer))
    print(json.dumps(build.stage_filter(ctx)))
    return 0


def _cmd_bios(a: argparse.Namespace) -> int:
    docs, probes = bios.generate(
        a.seed, a.docs, people=a.people, p_update=a.p_update, p_extra=a.p_extra
    )
    problems = bios.check_disjoint(docs, probes)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    out = Path(a.out)
    write_jsonl(out / "docs" / f"{bios.SOURCE}.jsonl", docs)
    write_jsonl(out / "probes" / "bios.jsonl", probes)
    print(json.dumps({"docs": len(docs), "probes": len(probes)}))
    return 0


def _cmd_chunk(a: argparse.Namespace) -> int:
    tok = get_tokenizer(a.tokenizer)
    modes = {k: s.chunk_mode for k, s in src.REGISTRY.items()}
    modes.update({k: "session" for k in a.session_sources})
    slices = {k: s.slice for k, s in src.REGISTRY.items()}
    n = write_jsonl(
        a.out,
        chunk.chunk_docs(
            _docs(a.docs),
            tok,
            sizes=_sizes(a.sizes),
            seed=a.seed,
            tolerance=a.tolerance,
            modes=modes,
            slices=slices,
            session_max=a.session_max,
        ),
    )
    print(json.dumps({"chunks": n, "tokenizer": tok.name}))
    return 0


def _cmd_decontam(a: argparse.Namespace) -> int:
    external = list(a.external or [])
    if not a.no_default_external:
        external += decontam.DEFAULT_EXTERNAL
    bench = (
        None
        if a.no_bench
        else {"users": a.bench_users, "sessions": a.bench_sessions, "seed": a.bench_seed}
    )
    ref = decontam.build_reference(
        n=a.n, bench=bench, bench_dirs=a.bench_dir or [], external=external
    )
    gen = set(a.generated_sources)
    kept, report = decontam.decontaminate_chunks(read_jsonl(a.chunks), ref, generated_sources=gen)
    write_jsonl(a.out, kept)
    full = {"reference": ref.summary(), **report}
    if a.probes:
        kp, prep = decontam.decontaminate_probes(read_jsonl(a.probes), ref, generated_sources=gen)
        write_jsonl(a.probes_out or Path(a.out).with_name("probes.decontam.jsonl"), kp)
        full["probes"] = prep["probes"]
        full["fatal"] = full["fatal"] + prep["fatal"]
    write_json(a.report, full)
    print(json.dumps({"kept": len(kept), "fatal": len(full["fatal"])}))
    return 1 if full["fatal"] else 0


def _cmd_probes(a: argparse.Namespace) -> int:
    chunks = list(read_jsonl(a.chunks))
    if a.max_chunks is not None:
        chunks = chunks[: a.max_chunks]
    generator: llm_probes.LLMProbeGenerator | None = llm_probes.LLMProbeGenerator(
        llm_probes.UnrunModel(a.model)
    )
    if not a.dry_run:
        if not a.base_url:
            print("probes: pass --dry-run, or --model and --base-url", file=sys.stderr)
            return 2
        cfg = {
            "generator": {"model": a.model, "base_url": a.base_url, "api_key_env": a.api_key_env}
        }
        generator = build.make_generator(cfg, Path(a.out) / "llm_cache.sqlite")
    summary = llm_probes.run_stage(chunks, Path(a.out), generator=generator, dry_run=a.dry_run)
    print(json.dumps(summary))
    return 0


def _cmd_episodes(a: argparse.Namespace) -> int:
    eps = episodes.build_episodes(
        read_jsonl(a.chunks),
        read_many(a.probes or []),
        seed=a.seed,
        k_max=a.k_max,
        max_context=a.max_context,
        p_mixed=a.p_mixed,
        max_probes=a.max_probes,
    )
    write_jsonl(a.out, eps)
    print(json.dumps({"episodes": len(eps), "probes": sum(len(e["probes"]) for e in eps)}))
    return 0


def _cmd_card(a: argparse.Namespace) -> int:
    tok = get_tokenizer(a.tokenizer)
    if a.dir:
        config = load_config(a.config) if a.config else None
        c = card.build_card(Path(a.dir), tok, config=config)
        rpath = Path(a.dir) / "contamination_report.json"
        if rpath.exists():
            rep = json.loads(rpath.read_text(encoding="utf-8"))
            c["contamination"] = {k: rep.get(k) for k in ("chunks", "probes", "fatal")}
    else:
        c = {"docs": card.doc_stats(_docs(a.docs), tok)}
    if a.out:
        write_json(a.out, c)
        Path(a.out).with_suffix(".md").write_text(card.render_markdown(c) + "\n", "utf-8")
    print(json.dumps(c, indent=2, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engram-corpus", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="run the stages from a config")
    b.add_argument("--config", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int)
    b.add_argument("--stages", help=f"comma list from {','.join(build.STAGES)}")
    b.add_argument("--cap", type=int, help="max documents per source (and bio docs)")
    b.add_argument("--target-tokens", type=float)
    b.add_argument("--tokenizer", help="override the config's tokenizer (e.g. whitespace)")
    b.add_argument("--dry-run", action="store_true", help="no model calls in the probe stage")
    b.add_argument("--resume", action="store_true", help="skip stages already completed")
    b.add_argument("--allow-ablation", action="store_true")
    b.set_defaults(func=_cmd_build)

    f = sub.add_parser("fetch", help="stream a capped number of documents per source")
    f.add_argument("--config")
    f.add_argument("--source", action="append", default=[], help="registry key (repeatable)")
    f.add_argument("--out", required=True)
    f.add_argument("--cap", type=int)
    f.add_argument("--target-tokens", type=float)
    f.add_argument("--tokenizer", default="whitespace")
    f.add_argument("--allow-ablation", action="store_true")
    f.set_defaults(func=_cmd_fetch)

    fl = sub.add_parser("filter", help="filter OUT/raw into OUT/docs")
    fl.add_argument("--config", required=True)
    fl.add_argument("--out", required=True)
    fl.add_argument("--tokenizer", default="whitespace")
    fl.set_defaults(func=_cmd_filter)

    bi = sub.add_parser("bios", help="generate synthetic biographies and exact probes")
    bi.add_argument("--out", required=True)
    bi.add_argument("--docs", type=int, default=10)
    bi.add_argument("--people", type=int, default=40)
    bi.add_argument("--seed", type=int, default=0)
    bi.add_argument("--p-update", type=float, default=0.3)
    bi.add_argument("--p-extra", type=float, default=0.5)
    bi.set_defaults(func=_cmd_bios)

    c = sub.add_parser("chunk", help="cut documents into 1/2/4K-token chunks")
    c.add_argument("--docs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--sizes", default="1024,2048,4096")
    c.add_argument("--tolerance", type=float, default=0.1)
    c.add_argument("--tokenizer", default="whitespace")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--session-max", type=int, default=4096)
    c.add_argument("--session-sources", nargs="*", default=[])
    c.set_defaults(func=_cmd_chunk)

    d = sub.add_parser("decontam", help="13-gram, template and canary filter against the evals")
    d.add_argument("--chunks", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--report", required=True)
    d.add_argument("--probes")
    d.add_argument("--probes-out")
    d.add_argument("--external", nargs="*", help="extra eval files or directories")
    d.add_argument("--no-default-external", action="store_true")
    d.add_argument("--bench-dir", nargs="*", help="saved engram-bench datasets")
    d.add_argument("--no-bench", action="store_true", help="do not generate engram-bench v0")
    d.add_argument("--bench-users", type=int, default=50)
    d.add_argument("--bench-sessions", type=int, default=30)
    d.add_argument("--bench-seed", type=int, default=0)
    d.add_argument("--generated-sources", nargs="*", default=[bios.SOURCE])
    d.add_argument("--n", type=int, default=13)
    d.set_defaults(func=_cmd_decontam)

    pr = sub.add_parser("probes", help="LLM probe generation for natural text (or --dry-run)")
    pr.add_argument("--chunks", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    pr.add_argument("--base-url")
    pr.add_argument("--api-key-env")
    pr.add_argument("--max-chunks", type=int)
    pr.set_defaults(func=_cmd_probes)

    e = sub.add_parser("episodes", help="assemble write-then-recall episodes")
    e.add_argument("--chunks", required=True)
    e.add_argument("--probes", nargs="*")
    e.add_argument("--out", required=True)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--k-max", type=int, default=128)
    e.add_argument("--max-context", type=int, default=131072)
    e.add_argument("--p-mixed", type=float, default=0.2)
    e.add_argument("--max-probes", type=int, default=16)
    e.set_defaults(func=_cmd_episodes)

    k = sub.add_parser("card", help="data-card statistics")
    g = k.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="a build directory")
    g.add_argument("--docs", nargs="+", help="document JSONL files")
    k.add_argument("--config", help="config, for the planned-vs-built mix table")
    k.add_argument("--tokenizer", default="whitespace")
    k.add_argument("--out", help="write card JSON here (and a .md next to it)")
    k.set_defaults(func=_cmd_card)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, OSError) as e:
        print(f"engram-corpus {args.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
