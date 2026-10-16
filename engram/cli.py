"""engram-bench command line.

    engram-bench generate --out data/bench-v0 --users 50 --sessions 30 --seed 0
    engram-bench run --bench data/bench-v0 --system b3 --model qwen3-8b \
        --base-url http://localhost:8000/v1 --out runs/b3-qwen3-8b
    engram-bench report runs/b3-qwen3-8b --floor runs/b0-qwen3-8b

Systems: b0 (no memory), b1 (full context), b2/b3 (retrieval; b2 = frontier model, b3 = base
model), and the controls oracle (reads gold answers) and leaky (no ownership gating). With
``--mock null|parrot`` any system runs on CPU with no model, for testing the plumbing.

Model specs for ``--model``:
    claude:<model>   Claude via the Claude Code CLI on your subscription (claude:claude-opus-5-5)
    codex:<model>    OpenAI via the Codex CLI on your ChatGPT subscription (codex:gpt-6.1-sol)
    <name>           an OpenAI-compatible endpoint at --base-url (vLLM; APIs via --api-key-env)
Responses are cached in ~/.cache/engram/responses.sqlite unless --no-cache, so interrupted runs
resume where they stopped.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from engram.bench import dataset
from engram.bench.runner import apply_floor, load_run, report, rescore, run, save_run
from engram.bench.writer import LLMWriter, TemplateWriter
from engram.models.cache import CachedModel
from engram.models.chat import ChatModel, NullModel, OpenAIChat, ParrotModel
from engram.models.subscription import ClaudeCLI, CodexCLI
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
    effort = getattr(args, "effort", "low")
    model: ChatModel
    if name.startswith("claude:"):
        model = ClaudeCLI(model=name.split(":", 1)[1], effort=effort)
    elif name.startswith("codex:"):
        model = CodexCLI(model=name.split(":", 1)[1], effort=effort)
    else:
        model = OpenAIChat(
            model=name,
            base_url=getattr(args, f"{prefix}base_url"),
            api_key_env=getattr(args, f"{prefix}api_key_env"),
        )
    return model if getattr(args, "no_cache", False) else CachedModel(model)


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
    elif args.system == "b4":
        from engram.systems.lora import LoRAConfig, LoRAMemory

        if not args.base_model:
            sys.exit("--base-model is required for b4")
        cfg = LoRAConfig(rank=args.lora_rank, epochs=args.lora_epochs, lr=args.lora_lr,
                         mode=args.lora_mode)  # fmt: skip
        system = LoRAMemory(args.base_model, cfg)
    else:
        model = _model(args)
        system = {
            "b0": lambda: NoMemory(model),
            "b1": lambda: FullContext(model, budget_chars=args.budget_chars),
            "b2": lambda: RetrievalMemory(model, k=args.k, label="b2", retriever=args.retriever),
            "b3": lambda: RetrievalMemory(model, k=args.k, label="b3", retriever=args.retriever),
            "leaky": lambda: LeakyRetrieval(model, k=args.k, retriever=args.retriever),
        }[args.system]()
    result = run(
        bench,
        system,
        leakage=not args.no_leakage,
        leak_targets_per_user=args.leak_targets,
        workers=args.workers,
        session_replies=not args.no_session_replies,
        progress=(lambda m: print(m, flush=True)) if args.verbose else None,
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


def cmd_import(args) -> None:
    from engram.bench import external

    loader = {"longmemeval": external.load_longmemeval, "locomo": external.load_locomo}
    bench = loader[args.source](Path(args.path), limit=args.limit)
    out = external.save(bench, Path(args.out))
    counts = ", ".join(f"{k}={v}" for k, v in external.summarize(bench).items())
    print(f"wrote {len(bench.users)} users to {out} ({counts}); "
          f"{len(external.judge_ids(bench))} probes need the LLM judge")  # fmt: skip


def cmd_judge(args) -> None:
    """Re-grade free-form and ambiguous answers with an LLM judge; everything else stays as is."""
    from concurrent.futures import ThreadPoolExecutor

    from engram.bench import external
    from engram.bench.judge import LLMJudge

    bench_dir = Path(args.bench)
    bench = (
        external.load(bench_dir)
        if (bench_dir / "external.json").exists()
        else dataset.load(bench_dir)
    )
    flagged = external.judge_ids(bench)
    probes = {p.id: p for p in bench.probes()}
    result = load_run(Path(args.run))
    judge = LLMJudge(_model(args))
    todo = [
        i for i, s in enumerate(result.scores)
        if (s.probe_id in flagged or s.ambiguous) and probes[s.probe_id].gold is not None
    ]  # fmt: skip

    def grade(i: int) -> tuple[int, bool | None]:
        s = result.scores[i]
        p = probes[s.probe_id]
        try:
            return i, judge.grade(p.question, p.gold or "", p.reject, s.answer)
        except Exception as e:  # keep the string-match verdict; re-running retries only these
            print(f"judge failed on {s.probe_id}: {e}", file=sys.stderr)
            return i, None

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, verdict in pool.map(grade, todo):
            if verdict is None:
                failed += 1
            else:
                result.scores[i] = dataclasses.replace(result.scores[i], correct=verdict)
    result.manifest["judge"] = {
        "model": judge.model.name,
        "judged": len(todo) - failed,
        "failed": failed,
    }
    save_run(result, Path(args.out or args.run))
    print(report(result.scores, result.leaks, f"{result.manifest.get('system_name')} (judged)"))


def cmd_calibrate(args) -> None:
    """Judge calibration (issue #2). ``export`` writes a CSV of answers to label by hand (column
    ``human``: 1 = correct, 0 = wrong); ``score`` compares those labels with the judge's."""
    import csv
    import random

    from engram.bench.judge import LLMJudge, cohen_kappa

    if args.action == "export":
        bench = dataset.load(Path(args.bench))
        probes = {p.id: p for p in bench.probes()}
        scores = [s for s in load_run(Path(args.run)).scores if probes[s.probe_id].gold]
        rng = random.Random(args.seed)
        sample = rng.sample(scores, min(args.n, len(scores)))
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["probe_id", "question", "gold", "reject", "answer", "human"])
            for s in sample:
                p = probes[s.probe_id]
                w.writerow([s.probe_id, p.question, p.gold, "; ".join(p.reject), s.answer, ""])
        print(f"wrote {len(sample)} rows to {args.csv}; fill the 'human' column with 1 or 0")
        return
    with open(args.csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human"].strip() in {"0", "1"}]
    if not rows:
        sys.exit("no labelled rows (human column must be 0 or 1)")
    judge = LLMJudge(_model(args))
    verdicts = [
        judge.grade(
            r["question"], r["gold"], [x for x in r["reject"].split("; ") if x], r["answer"]
        )
        for r in rows
    ]
    human = [r["human"].strip() == "1" for r in rows]
    agree = sum(a == b for a, b in zip(verdicts, human, strict=True)) / len(rows)
    print(f"n={len(rows)} agreement={agree:.3f} kappa={cohen_kappa(verdicts, human):.3f} "
          f"judge={judge.model.name}")  # fmt: skip


def cmd_compare(args) -> None:
    """Markdown table of accuracy (with 95% CI) by category, one column per run."""
    from engram.bench.scoring import summarise

    floor = load_run(Path(args.floor)).scores if args.floor else None
    cols, names = [], []
    for run_dir in args.runs:
        result = load_run(Path(run_dir))
        scores = apply_floor(result.scores, floor) if floor else result.scores
        cols.append({s.category: s for s in summarise(scores)})
        leaks = sum(c.leaked for c in result.leaks)
        suffix = f" (leaks {leaks}/{len(result.leaks)})" if result.leaks else ""
        names.append(f"{result.manifest.get('system_name', run_dir)}{suffix}")
    cats = sorted({c for col in cols for c in col}, key=lambda c: (c == "ALL", c))
    print("| category | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for cat in cats:
        cells = []
        for col in cols:
            s = col.get(cat)
            cells.append(
                f"{s.accuracy:.2f} [{s.ci_low:.2f}, {s.ci_high:.2f}] (n={s.n})" if s else "–"
            )
        print(f"| {cat} | " + " | ".join(cells) + " |")


def cmd_rescore(args) -> None:
    bench = dataset.load(Path(args.bench))
    result = rescore(load_run(Path(args.run)), bench)
    save_run(result, Path(args.run))
    print(report(result.scores, result.leaks, f"{result.manifest.get('system_name')} (rescored)"))


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
    g.add_argument("--no-cache", action="store_true")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run a system through the benchmark")
    r.add_argument("--bench", required=True)
    r.add_argument(
        "--system", choices=["b0", "b1", "b2", "b3", "b4", "oracle", "leaky"], required=True
    )
    r.add_argument("--model")
    r.add_argument("--mock", choices=["null", "parrot"])
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--api-key-env", help="name of the env var holding the API key")
    r.add_argument("--k", type=int, default=5, help="retrieval depth for b2/b3")
    r.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="b2/b3 retriever; dense/hybrid need the ml extra (use hybrid for reported numbers)",
    )
    r.add_argument("--budget-chars", type=int, default=400_000, help="b1 context budget")
    r.add_argument("--limit-users", type=int)
    r.add_argument("--workers", type=int, default=1, help="users run in parallel")
    r.add_argument(
        "--no-session-replies",
        action="store_true",
        help="log session turns without generating replies (valid for all baselines)",
    )
    r.add_argument("--leak-targets", type=int, default=3, help="other users probed per user")
    r.add_argument("--base-model", help="b4: Hugging Face id of the frozen base model")
    r.add_argument("--lora-rank", type=int, default=16)
    r.add_argument("--lora-epochs", type=int, default=3)
    r.add_argument("--lora-lr", type=float, default=2e-4)
    r.add_argument("--lora-mode", choices=["cumulative", "incremental"], default="cumulative")
    r.add_argument("--effort", default="low", help="reasoning effort for claude:/codex: models")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--no-leakage", action="store_true")
    r.add_argument("--out", required=True)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="summarise a saved run")
    p.add_argument("run")
    p.add_argument("--floor", help="a B0 run directory; drops probes B0 already answers")
    p.set_defaults(func=cmd_report)

    im = sub.add_parser("import", help="convert LongMemEval or LoCoMo into engram-bench format")
    im.add_argument("source", choices=["longmemeval", "locomo"])
    im.add_argument("path", help="the downloaded JSON file")
    im.add_argument("--out", required=True)
    im.add_argument("--limit", type=int, help="convert only the first N instances")
    im.set_defaults(func=cmd_import)

    j = sub.add_parser("judge", help="grade free-form/ambiguous answers with an LLM judge")
    j.add_argument("run")
    j.add_argument("--bench", required=True)
    j.add_argument("--model", required=True, help="judge model spec, e.g. claude:claude-opus-5-5")
    j.add_argument("--mock", choices=["null", "parrot"])
    j.add_argument("--base-url", default="http://localhost:8000/v1")
    j.add_argument("--api-key-env")
    j.add_argument("--effort", default="low")
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--out", help="write the judged run here instead of in place")
    j.set_defaults(func=cmd_judge)

    c = sub.add_parser("calibrate", help="export answers to label, or score the judge vs labels")
    c.add_argument("action", choices=["export", "score"])
    c.add_argument("csv")
    c.add_argument("--run", help="export: run directory to sample answers from")
    c.add_argument("--bench", help="export: the run's benchmark directory")
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--model", help="score: judge model spec")
    c.add_argument("--mock", choices=["null", "parrot"])
    c.add_argument("--base-url", default="http://localhost:8000/v1")
    c.add_argument("--api-key-env")
    c.add_argument("--effort", default="low")
    c.set_defaults(func=cmd_calibrate)

    cp = sub.add_parser("compare", help="markdown table comparing several runs")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("--floor", help="a B0 run; drops probes it already answers")
    cp.set_defaults(func=cmd_compare)

    rs = sub.add_parser("rescore", help="re-score a saved run's answers with current rules")
    rs.add_argument("run")
    rs.add_argument("--bench", required=True)
    rs.set_defaults(func=cmd_rescore)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
