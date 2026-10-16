"""engram-bench command line.

    engram-bench generate --out data/bench-v0 --users 50 --sessions 30 --seed 0
    engram-bench run --bench data/bench-v0 --system b3 --model qwen3-8b \
        --base-url http://localhost:8000/v1 --out runs/b3-qwen3-8b
    engram-bench report runs/b3-qwen3-8b --floor runs/b0-qwen3-8b

Systems: b0 (no memory), b1 (full context), b2/b3 (retrieval; b2 = frontier model, b3 = base
model), and the controls oracle (reads gold answers) and leaky (no ownership gating). With
``--mock null|parrot`` any system runs on CPU with no model, for testing the plumbing.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from engram.bench import dataset
from engram.bench.runner import apply_floor, load_run, report, run, save_run
from engram.bench.writer import LLMWriter, TemplateWriter
from engram.models.chat import ChatModel, NullModel, OpenAIChat, ParrotModel
from engram.systems.baselines import (
    FullContext,
    LeakyRetrieval,
    NoMemory,
    OracleMemory,
    RetrievalMemory,
)


def _model(args, prefix: str = "") -> ChatModel:
    mock = getattr(args, f"{prefix}mock", None)
    if mock == "null":
        return NullModel()
    if mock == "parrot":
        return ParrotModel()
    name = getattr(args, f"{prefix}model")
    if not name:
        sys.exit(f"--{prefix.replace('_', '-')}model or --{prefix.replace('_', '-')}mock required")
    return OpenAIChat(
        model=name,
        base_url=getattr(args, f"{prefix}base_url"),
        api_key_env=getattr(args, f"{prefix}api_key_env"),
    )


def cmd_generate(args) -> None:
    config = dataset.BenchConfig(
        users=args.users, sessions=args.sessions, seed=args.seed, sleep_every=args.sleep_every
    )
    writer = TemplateWriter() if args.writer == "template" else LLMWriter(_model(args, "writer_"))
    bench = dataset.generate(config, writer)
    out = dataset.save(bench, Path(args.out))
    n = sum(len(u.probes) for u in bench.users)
    print(f"wrote {len(bench.users)} users, {n} probes to {out} (writer: {bench.writer})")
    if isinstance(writer, LLMWriter):
        print(f"LLM rewrites that fell back to templates: {writer.fallbacks}")


def cmd_run(args) -> None:
    bench = dataset.load(Path(args.bench))
    if args.limit_users:
        bench = dataclasses.replace(bench, users=bench.users[: args.limit_users])
    if args.system == "oracle":
        system = OracleMemory(bench)
    else:
        model = _model(args)
        system = {
            "b0": lambda: NoMemory(model),
            "b1": lambda: FullContext(model, budget_chars=args.budget_chars),
            "b2": lambda: RetrievalMemory(model, k=args.k, label="b2"),
            "b3": lambda: RetrievalMemory(model, k=args.k, label="b3"),
            "leaky": lambda: LeakyRetrieval(model, k=args.k),
        }[args.system]()
    result = run(
        bench, system, leakage=not args.no_leakage, progress=print if args.verbose else None
    )
    out = save_run(result, Path(args.out))
    print(report(result.scores, result.leaks, f"{system.name} on {args.bench}"))
    print(f"\nsaved to {out}")


def cmd_report(args) -> None:
    result = load_run(Path(args.run))
    scores = result.scores
    title = f"{result.manifest.get('system_name', args.run)}"
    if args.floor:
        floor = load_run(Path(args.floor)).scores
        before = len(scores)
        scores = apply_floor(scores, floor)
        title += f" (floor applied: dropped {before - len(scores)} probes the no-memory run solved)"
    print(report(scores, result.leaks, title))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="engram-bench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a benchmark dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--users", type=int, default=50)
    g.add_argument("--sessions", type=int, default=30)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--sleep-every", type=int, default=3)
    g.add_argument("--writer", choices=["template", "llm"], default="template")
    g.add_argument("--writer-model")
    g.add_argument("--writer-mock", choices=["null", "parrot"])
    g.add_argument("--writer-base-url", default="http://localhost:8000/v1")
    g.add_argument("--writer-api-key-env")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run a system through the benchmark")
    r.add_argument("--bench", required=True)
    r.add_argument("--system", choices=["b0", "b1", "b2", "b3", "oracle", "leaky"], required=True)
    r.add_argument("--model")
    r.add_argument("--mock", choices=["null", "parrot"])
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--api-key-env", help="name of the env var holding the API key")
    r.add_argument("--k", type=int, default=5, help="retrieval depth for b2/b3")
    r.add_argument("--budget-chars", type=int, default=400_000, help="b1 context budget")
    r.add_argument("--limit-users", type=int)
    r.add_argument("--no-leakage", action="store_true")
    r.add_argument("--out", required=True)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="summarise a saved run")
    p.add_argument("run")
    p.add_argument("--floor", help="a B0 run directory; drops probes B0 already answers")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
