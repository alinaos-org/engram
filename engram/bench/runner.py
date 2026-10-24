"""Run a memory system through the engram-bench protocol and score it.

For each user: load their module; for each session, play the scripted turns, wipe the context,
ask the probes due before sleep, sleep if scheduled, ask the probes due after sleep. After every
user has been run, the leakage harness probes across users.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import platform
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from engram import __version__
from engram.bench.dataset import Benchmark, content_hash
from engram.bench.leakage import LeakCheck, run_leakage
from engram.bench.schema import Phase, Probe
from engram.bench.scoring import Score, format_table, score, summarise
from engram.bench.seal import PER_PROBE, refuse_in_repo
from engram.models.cache import usage
from engram.systems.base import MemorySystem


@dataclass
class RunResult:
    scores: list[Score] = field(default_factory=list)
    leaks: list[LeakCheck] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


def start(system: MemorySystem, *, write: bool, day: int | None = None) -> None:
    """Start a session, passing the simulated day to systems whose ``start_session`` takes it."""
    try:
        accepts_day = "day" in inspect.signature(system.start_session).parameters
    except (TypeError, ValueError):
        accepts_day = False
    if accepts_day:
        system.start_session(write=write, day=day)
    else:
        system.start_session(write=write)


def ask(system: MemorySystem, question: str, day: int | None = None) -> str:
    """Probes run in their own session and must never be written to memory."""
    start(system, write=False, day=day)
    answer = system.respond(question)
    system.end_session()
    return answer


def _git_commit() -> str | None:
    """HEAD of the engram checkout this code runs from, not of the cwd's repository, with
    ``-dirty`` for uncommitted changes (as in ``engram.corpus.build``); None for an installed
    copy. ``GIT_*`` variables (``GIT_DIR`` in a hook, say) are dropped: they override ``-C``."""
    root = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def git(*args: str) -> str:
        cmd = ["git", "-C", str(root), *args]
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5, env=env
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return None  # an installed copy inside some other repository
        return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _play_user(system: MemorySystem, user, sleeps: set[int], session_replies: bool) -> list[Score]:
    """Run one user's full timeline through ``system`` and score that user's probes."""
    due: dict[tuple[int, Phase], list[Probe]] = defaultdict(list)
    for p in user.probes:
        due[(p.ask_after_session, p.phase)].append(p)
    clock = getattr(system, "set_clock", None)
    observe = None if session_replies else getattr(system, "observe", None)
    scores: list[Score] = []

    def ask_due(s: int, phase: Phase) -> None:
        if clock:
            clock(s, phase.value)
        for p in due.get((s, phase), []):
            scores.append(score(p, ask(system, p.question, user.sessions[s].day)))

    system.load_user(user.id)
    for session in user.sessions:
        start(system, write=not session.incognito, day=session.day)
        for turn in session.turns:
            if observe:
                observe(turn.text)
            else:
                system.respond(turn.text)
        system.end_session()
        ask_due(session.index, Phase.PRE_SLEEP)
        if session.index in sleeps:
            system.sleep()
            ask_due(session.index, Phase.POST_SLEEP)
    system.unload_user()
    return scores


def _replay(system: MemorySystem, bench: Benchmark, sleeps: set[int]) -> None:
    """Rebuild every user's memory in one instance without model calls (needs ``observe``)."""
    for user in bench.users:
        system.load_user(user.id)
        for session in user.sessions:
            start(system, write=not session.incognito, day=session.day)
            for turn in session.turns:
                system.observe(turn.text)  # type: ignore[attr-defined]
            system.end_session()
            if session.index in sleeps:
                system.sleep()
        system.unload_user()


def run(
    bench: Benchmark,
    system: MemorySystem,
    *,
    leakage: bool = True,
    leak_targets_per_user: int = 3,
    workers: int = 1,
    session_replies: bool = True,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Run ``system`` through the benchmark.

    With ``workers > 1`` each user runs on a fresh instance from ``system.spawn()`` in its own
    thread. The leakage pass then needs every user's memory in one instance, so it is rebuilt with
    ``observe`` (no model calls) and probed through ``fork_readonly`` views in parallel. Systems
    without those methods run with ``workers=1``, where one instance carries every user.

    ``session_replies=False`` feeds session turns through ``observe`` instead of ``respond``. Only
    valid for systems whose memory does not depend on their own replies (all current baselines).
    """
    started = time.time()
    commit = _git_commit()  # the code that runs, not whatever is checked out when it ends
    sleeps = set(bench.sleeps)
    result = RunResult()
    parallel = workers > 1
    if parallel and not (hasattr(system, "spawn") and hasattr(system, "observe")):
        raise ValueError(f"{system.name} does not support parallel runs; use workers=1")
    if not session_replies and not hasattr(system, "observe"):
        raise ValueError(f"{system.name} has no observe(); it needs session_replies=True")

    done = 0

    def report_progress(user) -> None:
        nonlocal done
        done += 1
        if progress:
            rate = (time.time() - started) / done
            left = rate * (len(bench.users) - done)
            progress(f"{done}/{len(bench.users)} users ({user.id}); ~{left / 60:.0f} min left")

    if parallel:
        with ThreadPoolExecutor(workers) as pool:
            futures = {
                pool.submit(_play_user, system.spawn(), u, sleeps, session_replies): u
                for u in bench.users
            }
            by_user: dict[str, list[Score]] = {}
            for fut in as_completed(futures):
                by_user[futures[fut].id] = fut.result()
                report_progress(futures[fut])
        for user in bench.users:
            result.scores.extend(by_user[user.id])
    else:
        for user in bench.users:
            result.scores.extend(_play_user(system, user, sleeps, session_replies))
            report_progress(user)

    if leakage:
        if parallel:
            _replay(system, bench, sleeps)
        result.leaks = run_leakage(
            bench, system, targets_per_user=leak_targets_per_user, workers=workers
        )

    bench_info = {"config": asdict(bench.config), "writer": bench.writer}
    if bench.sealed:  # never record a sealed split's seed; bench_sha256 ties the run to it
        bench_info["config"]["seed"] = None
        bench_info["sealed"] = dict(bench.sealed)
    result.manifest = {
        "engram_version": __version__,
        "git_commit": commit,
        "system": system.describe(),
        "system_name": system.name,
        "bench": bench_info,
        "bench_sha256": content_hash(bench),
        "n_users": len(bench.users),
        "workers": workers,
        "session_replies": session_replies,
        "python": platform.python_version(),
        "host": platform.node(),
        "wall_clock_s": round(time.time() - started, 2),
        "n_scores": len(result.scores),
        "n_leak_checks": len(result.leaks),
    }
    model = getattr(system, "model", None)
    if model is not None and hasattr(model, "hits"):  # kept for older readers; see "usage"
        result.manifest["cache"] = {"hits": model.hits, "misses": model.misses}
    # What each model role drew from its backend; ``judge`` adds its own role to this.
    result.manifest["usage"] = {"answer": usage(model)} if hasattr(model, "complete") else {}
    return result


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def scrub(text: str) -> tuple[str, int]:
    """Redact email addresses. Synthetic users have none, so any email in an answer came from the
    model's own context (the Claude CLI injects the operator's account email) and must never be
    written to results that may be published."""
    return _EMAIL.subn("[redacted-email]", text)


def save_run(result: RunResult, out: Path, *, allow_in_repo: bool = False) -> Path:
    if result.manifest.get("bench", {}).get("sealed"):
        refuse_in_repo(out, PER_PROBE, allow=allow_in_repo)
    out.mkdir(parents=True, exist_ok=True)
    redacted = 0
    for coll in (result.scores, result.leaks):
        for i, item in enumerate(coll):
            clean, n = scrub(item.answer)
            if n:
                coll[i] = dataclasses.replace(item, answer=clean)
                redacted += n
    result.manifest["redacted_emails"] = result.manifest.get("redacted_emails", 0) + redacted
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for s in result.scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    with (out / "leakage.jsonl").open("w", encoding="utf-8") as f:
        for leak in result.leaks:
            f.write(json.dumps(asdict(leak), ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(result.manifest, indent=2) + "\n")
    (out / "summary.txt").write_text(report(result.scores, result.leaks) + "\n")
    return out


def load_run(path: Path) -> RunResult:
    def lines(name: str) -> list[dict]:
        p = path / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    return RunResult(
        scores=[Score(**d) for d in lines("results.jsonl")],
        leaks=[LeakCheck(**d) for d in lines("leakage.jsonl")],
        manifest=json.loads((path / "manifest.json").read_text()),
    )


def rescore(result: RunResult, bench: Benchmark) -> RunResult:
    """Re-score saved answers with the current scoring rules. No model calls."""
    probes = {p.id: p for p in bench.probes()}
    scores = [score(probes[s.probe_id], s.answer) for s in result.scores]
    return RunResult(scores, result.leaks, {**result.manifest, "rescored": True})


def apply_floor(scores: list[Score], floor: list[Score]) -> list[Score]:
    """Leak-verified floor (MERIT): drop answerable probes that the no-memory baseline already
    gets right, since a correct answer there cannot be evidence of memory."""
    known = {s.probe_id for s in floor if s.correct and s.category not in _NO_GOLD}
    return [s for s in scores if s.probe_id not in known]


_NO_GOLD = {"abstention", "forget", "incognito"}


def _leaks_by_kind(leaks: list[LeakCheck]) -> dict[str, list[LeakCheck]]:
    by_kind: dict[str, list[LeakCheck]] = defaultdict(list)
    for leak in leaks:
        by_kind[leak.kind].append(leak)
    return dict(sorted(by_kind.items()))


def summary_json(
    manifest: dict, scores: list[Score], leaks: list[LeakCheck], floor: dict | None = None
) -> dict:
    """The aggregates of ``report`` as one JSON object. It holds no probe ids, questions,
    answers or values, so it is safe to commit for a sealed split."""
    return {
        "manifest": manifest,
        "floor": floor,
        "by_category": [asdict(s) for s in summarise(scores)],
        "by_timing": [asdict(s) for s in summarise(scores, by="timing")],
        "leakage": {
            kind: {"checks": len(group), "leaks": sum(g.leaked for g in group)}
            for kind, group in _leaks_by_kind(leaks).items()
        },
    }


def report(scores: list[Score], leaks: list[LeakCheck], title: str = "") -> str:
    parts = [format_table(summarise(scores), title or "Accuracy by category")]
    parts.append(format_table(summarise(scores, by="timing"), "\nAccuracy by probe timing"))
    if leaks:
        parts.append("\nCross-user leakage (target: 0)")
        for kind, group in _leaks_by_kind(leaks).items():
            parts.append(f"{kind:<14} checks={len(group):>5} leaks={sum(g.leaked for g in group)}")
    return "\n".join(parts)
