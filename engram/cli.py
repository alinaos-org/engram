"""engram-bench command line.

    engram-bench generate --out data/bench-v0 --users 50 --sessions 30 --seed 0
    engram-bench run --bench data/bench-v0 --system b3 --model qwen3-8b \
        --base-url http://localhost:8000/v1 --out runs/b3-qwen3-8b
    engram-bench report runs/b3-qwen3-8b --floor runs/b0-qwen3-8b
    engram-bench generate --seal ~/.local/share/engram/sealed/bench-v1-test.seed.json \
        --out ~/.local/share/engram/sealed/bench-v1-test --users 50 --sessions 30
    engram-bench verify-seal ~/.local/share/engram/sealed/bench-v1-test \
        --seed-file ~/.local/share/engram/sealed/bench-v1-test.seed.json

Systems: b0 (no memory), b1 (full context), b2/b3 (retrieval; b2 = frontier model, b3 = base
model), and the controls oracle (reads gold answers) and leaky (no ownership gating). With
``--mock null|parrot`` any system runs on CPU with no model, for testing the plumbing.

Model specs for ``--model``:
    claude:<model>   Claude via the Claude Code CLI on your subscription (claude:claude-opus-5-5)
    codex:<model>    OpenAI via the Codex CLI on your ChatGPT subscription (codex:gpt-6.1-sol)
    <name>           an OpenAI-compatible endpoint at --base-url (vLLM; APIs via --api-key-env)
Responses are cached in ~/.cache/engram/responses.sqlite unless --no-cache, so interrupted runs
resume where they stopped. Run and dataset manifests record each model role's calls, cache hits,
CLI attempts (retries included) and subscription pool under "usage".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
from pathlib import Path

from engram.bench import dataset, seal
from engram.bench.runner import (
    _git_commit,
    apply_floor,
    load_run,
    report,
    rescore,
    run,
    save_run,
    summary_json,
)
from engram.bench.writer import LLMWriter, LongWriter, TemplateWriter
from engram.models.cache import CachedModel, usage
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
    if getattr(args, "no_cache", False):
        return CachedModel(model, path=None)  # stores nothing, still counts calls
    return CachedModel(model)


def _refuse_overwrite(out: Path, sealing: bool) -> None:
    """Like its seed file, a sealed split is never written over: its seed rebuilds the plan but
    not LLM-written sessions. ``--seal`` refuses any existing dataset; plain ``generate`` refuses
    only a sealed one."""
    manifest = out / "manifest.json"
    if sealing and (manifest.exists() or (out / "users.jsonl").exists()):
        raise ValueError(f"{out} already holds a dataset; generate --seal never overwrites one")
    try:
        sealed = "sealed" in json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        sealed = False
    if sealed:
        raise ValueError(f"{out} holds a sealed split, which is never overwritten")


def cmd_generate(args) -> None:
    if args.seal is not None:
        seal.refuse_in_repo(Path(args.out), seal.PER_PROBE, allow=args.allow_in_repo)
    _refuse_overwrite(Path(args.out), args.seal is not None)
    config = dataset.BenchConfig(
        users=args.users,
        sessions=args.sessions,
        seed=None if args.seal is not None else 0 if args.seed is None else args.seed,
        sleep_every=args.sleep_every,
        application_probes=args.application_probes,
    )
    if args.writer == "template":
        writer = TemplateWriter()
    elif args.writer == "long":
        writer = LongWriter(_model(args, "writer_"), workers=args.writer_workers)
    else:
        writer = LLMWriter(_model(args, "writer_"))
    record = sealed = None
    if args.seal is not None:
        # The seed file is written, and the commit read, before the first writer call: a long
        # generation never loses the seed, and a checkout during it cannot change the commit.
        record = seal.new_record(seal.spec_sha256(dataclasses.asdict(config), writer.name))
        seal.write_seed_file(Path(args.seal), record, dataset_dir=Path(args.out))
        config = dataclasses.replace(config, seed=seal.seed_int(record))
        sealed = {
            **seal.sealed_block(record),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
        }
    bench = dataset.generate(config, writer)
    bench.sealed = sealed
    if isinstance(writer, LLMWriter | LongWriter):
        drew = {**usage(writer.model), "fallbacks": writer.fallbacks}
        if isinstance(writer, LongWriter):
            drew["failed_sessions"] = writer.failed_sessions
        bench.usage = {"writer": drew}
    out = dataset.save(bench, Path(args.out))
    n = sum(len(u.probes) for u in bench.users)
    print(f"wrote {len(bench.users)} users, {n} probes to {out} (writer: {bench.writer})")
    if record is not None:
        print(f"sealed: seed in {args.seal} (keep it private until the reveal); "
              f"commitment {record['seed_commitment']}")  # fmt: skip
    if isinstance(writer, LLMWriter | LongWriter):
        print(f"LLM rewrites that fell back to templates: {writer.fallbacks}")
    if isinstance(writer, LongWriter):
        print(f"sessions the model failed to write (filler used): {writer.failed_sessions}")


def cmd_run(args) -> None:
    bench = dataset.load(Path(args.bench))
    if bench.sealed:  # refuse before the first model call, not after the run
        seal.refuse_in_repo(Path(args.out), seal.PER_PROBE, allow=args.allow_in_repo)
    if args.limit_users:
        bench = dataclasses.replace(bench, users=bench.users[: args.limit_users])
    if args.system == "oracle":
        system = OracleMemory(bench)
    elif args.system == "b5":
        from engram.systems.mem0_memory import Mem0Memory

        system = Mem0Memory(_model(args), k=args.k)
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
    membership = None
    if args.membership:
        # Membership inference reads the adapters, and the leakage pass deletes users, so run the
        # protocol without leakage, attack the adapters, then run the leakage pass.
        if not hasattr(system, "sequence_losses"):
            sys.exit("--membership needs a system with per-user adapters (b4)")
        from engram.bench.leakage import run_leakage
        from engram.bench.membership import evaluate, format_report

        result = run(
            bench, system, leakage=False, workers=args.workers,
            session_replies=not args.no_session_replies,
            progress=(lambda m: print(m, flush=True)) if args.verbose else None,
        )  # fmt: skip
        membership = evaluate(system, bench)
        print(format_report(membership))
        if not args.no_leakage:
            result.leaks = run_leakage(bench, system, targets_per_user=args.leak_targets)
            result.manifest["n_leak_checks"] = len(result.leaks)
    else:
        result = run(
            bench,
            system,
            leakage=not args.no_leakage,
            leak_targets_per_user=args.leak_targets,
            workers=args.workers,
            session_replies=not args.no_session_replies,
            progress=(lambda m: print(m, flush=True)) if args.verbose else None,
        )
    out = save_run(result, Path(args.out), allow_in_repo=args.allow_in_repo)
    if membership is not None:
        from engram.bench.membership import format_report as _fmt

        summary = {
            "report": _fmt(membership, per_user=False),
            "pooled": {
                g: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                for g, v in getattr(membership, "pooled", {}).items()
            },
        }
        (out / "membership.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(report(result.scores, result.leaks, f"{system.name} on {args.bench}"))
    print(f"\nsaved to {out}")


def cmd_report(args) -> None:
    result = load_run(Path(args.run))
    scores = result.scores
    title = f"{result.manifest.get('system_name', args.run)}"
    floor_info = None
    if args.floor:
        floor = load_run(Path(args.floor))
        before = len(scores)
        scores = apply_floor(scores, floor.scores)
        floor_info = {
            "system_name": floor.manifest.get("system_name"),
            "bench_sha256": floor.manifest.get("bench_sha256"),
            "dropped": before - len(scores),
        }
        title += f" (floor applied: dropped {before - len(scores)} probes the no-memory run solved)"
    print(report(scores, result.leaks, title))
    if args.json:
        summary = summary_json(result.manifest, scores, result.leaks, floor_info)
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


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
    _mark_sealed(result, bench)
    if result.manifest.get("bench", {}).get("sealed"):
        seal.refuse_in_repo(Path(args.out or args.run), seal.PER_PROBE, allow=args.allow_in_repo)
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
    # Like "judge", this describes the latest judge pass on the run. A run saved before "usage"
    # existed gets "answer": null (not recorded), not a missing key, which would mean no calls.
    result.manifest.setdefault("usage", {"answer": None})
    result.manifest["usage"]["judge"] = {**usage(judge.model), "failed": failed}
    save_run(result, Path(args.out or args.run), allow_in_repo=args.allow_in_repo)
    print(report(result.scores, result.leaks, f"{result.manifest.get('system_name')} (judged)"))


def cmd_calibrate(args) -> None:
    """Judge calibration (issue #2). ``export`` writes a CSV of answers to label by hand (column
    ``human``: 1 = correct, 0 = wrong); ``score`` compares those labels with the judge's."""
    import csv
    import random

    from engram.bench.judge import LLMJudge, cohen_kappa

    if args.action == "export":
        bench = dataset.load(Path(args.bench))
        if bench.sealed:
            seal.refuse_in_repo(Path(args.csv), seal.PER_PROBE, allow=args.allow_in_repo)
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
    print(f"usage: {json.dumps({'judge': usage(judge.model)})}")  # no manifest records it


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


def _mark_sealed(result, bench) -> None:
    """Copy a sealed bench's block into the run manifest, so the guards (``save_run``'s too) fire
    for a run saved without one, as engram wrote runs before sealed splits."""
    if bench.sealed:
        result.manifest.setdefault("bench", {})["sealed"] = dict(bench.sealed)


def cmd_rescore(args) -> None:
    bench = dataset.load(Path(args.bench))
    if bench.sealed:
        seal.refuse_in_repo(Path(args.run), seal.PER_PROBE, allow=args.allow_in_repo)
    result = load_run(Path(args.run))
    _mark_sealed(result, bench)
    result = rescore(result, bench)
    save_run(result, Path(args.run), allow_in_repo=args.allow_in_repo)
    print(report(result.scores, result.leaks, f"{result.manifest.get('system_name')} (rescored)"))


def cmd_verify_seal(args) -> None:
    # Each line is printed as its check passes, so a failure still shows what passed before it.
    dataset.verify_seal(Path(args.path), Path(args.seed_file), report=print)


ALLOW_IN_REPO = "write sealed per-probe content inside a git work tree anyway"


def _path(arg: str) -> Path:
    """A path option with ``~`` expanded. Shells leave ``--out=~/x`` as typed, and the work-tree
    guard must check the path that is written, not ``./~/x``. An empty value is an error."""
    if not arg.strip():
        raise argparse.ArgumentTypeError("needs a path")
    try:
        return Path(arg).expanduser()
    except RuntimeError as e:  # "~nosuchuser/x", or "~" with no home directory
        raise argparse.ArgumentTypeError(str(e)) from None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="engram-bench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a benchmark dataset")
    g.add_argument("--out", required=True, type=_path)
    g.add_argument("--users", type=int, default=50)
    g.add_argument("--sessions", type=int, default=30)
    seeding = g.add_mutually_exclusive_group()
    # default=None, not 0: Python 3.11's group misses "--seed 0 --seal" when 0 is the default.
    seeding.add_argument("--seed", type=int, default=None, help="default 0")
    seeding.add_argument(
        "--seal",
        type=_path,
        metavar="SEED_FILE",
        help="sealed split: draw a random seed into SEED_FILE (outside every repository) and "
        "publish only a commitment to it",
    )
    g.add_argument(
        "--allow-in-repo",
        action="store_true",
        help="--seal: write the dataset (--out) inside a git work tree anyway; never the seed file",
    )
    g.add_argument("--sleep-every", type=int, default=3)
    g.add_argument("--writer", choices=["template", "llm", "long"], default="template")
    g.add_argument("--writer-workers", type=int, default=4, help="long writer: parallel sessions")
    g.add_argument(
        "--application-probes", action="store_true", help="v1 preference-application probes"
    )
    g.add_argument("--writer-model")
    g.add_argument("--writer-mock", choices=["null", "parrot"])
    g.add_argument("--writer-base-url", default="http://localhost:8000/v1")
    g.add_argument("--writer-api-key-env")
    g.add_argument("--no-cache", action="store_true")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run a system through the benchmark")
    r.add_argument("--bench", required=True, type=_path)
    r.add_argument(
        "--system", choices=["b0", "b1", "b2", "b3", "b4", "b5", "oracle", "leaky"], required=True
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
    r.add_argument(
        "--membership",
        action="store_true",
        help="b4: reference-calibrated membership inference on the adapters (before leakage)",
    )
    r.add_argument("--lora-rank", type=int, default=16)
    r.add_argument("--lora-epochs", type=int, default=3)
    r.add_argument("--lora-lr", type=float, default=2e-4)
    r.add_argument("--lora-mode", choices=["cumulative", "incremental"], default="cumulative")
    r.add_argument("--effort", default="low", help="reasoning effort for claude:/codex: models")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--no-leakage", action="store_true")
    r.add_argument("--out", required=True, type=_path)
    r.add_argument("--allow-in-repo", action="store_true", help=ALLOW_IN_REPO)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="summarise a saved run")
    p.add_argument("run")
    p.add_argument("--floor", help="a B0 run directory; drops probes B0 already answers")
    p.add_argument("--json", help="also write the aggregates here (no per-probe content)")
    p.set_defaults(func=cmd_report)

    im = sub.add_parser("import", help="convert LongMemEval or LoCoMo into engram-bench format")
    im.add_argument("source", choices=["longmemeval", "locomo"])
    im.add_argument("path", help="the downloaded JSON file")
    im.add_argument("--out", required=True)
    im.add_argument("--limit", type=int, help="convert only the first N instances")
    im.set_defaults(func=cmd_import)

    j = sub.add_parser("judge", help="grade free-form/ambiguous answers with an LLM judge")
    j.add_argument("run", type=_path)
    j.add_argument("--bench", required=True, type=_path)
    j.add_argument("--model", required=True, help="judge model spec, e.g. claude:claude-opus-5-5")
    j.add_argument("--mock", choices=["null", "parrot"])
    j.add_argument("--base-url", default="http://localhost:8000/v1")
    j.add_argument("--api-key-env")
    j.add_argument("--effort", default="low")
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--out", type=_path, help="write the judged run here instead of in place")
    j.add_argument("--allow-in-repo", action="store_true", help=ALLOW_IN_REPO)
    j.set_defaults(func=cmd_judge)

    c = sub.add_parser("calibrate", help="export answers to label, or score the judge vs labels")
    c.add_argument("action", choices=["export", "score"])
    c.add_argument("csv", type=_path)
    c.add_argument("--run", type=_path, help="export: run directory to sample answers from")
    c.add_argument("--bench", type=_path, help="export: the run's benchmark directory")
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--model", help="score: judge model spec")
    c.add_argument("--mock", choices=["null", "parrot"])
    c.add_argument("--base-url", default="http://localhost:8000/v1")
    c.add_argument("--api-key-env")
    c.add_argument("--effort", default="low")
    c.add_argument("--allow-in-repo", action="store_true", help=ALLOW_IN_REPO)
    c.set_defaults(func=cmd_calibrate)

    cp = sub.add_parser("compare", help="markdown table comparing several runs")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("--floor", help="a B0 run; drops probes it already answers")
    cp.set_defaults(func=cmd_compare)

    rs = sub.add_parser("rescore", help="re-score a saved run's answers with current rules")
    rs.add_argument("run", type=_path)
    rs.add_argument("--bench", required=True, type=_path)
    rs.add_argument("--allow-in-repo", action="store_true", help=ALLOW_IN_REPO)
    rs.set_defaults(func=cmd_rescore)

    vs = sub.add_parser("verify-seal", help="check a sealed split against its revealed seed")
    vs.add_argument("path", type=_path, help="a dataset directory, or a bare manifest file")
    vs.add_argument("--seed-file", required=True, type=_path)
    vs.set_defaults(func=cmd_verify_seal)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
