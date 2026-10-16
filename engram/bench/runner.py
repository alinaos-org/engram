"""Run a memory system through the engram-bench protocol and score it.

For each user: load their module; for each session, play the scripted turns, wipe the context,
ask the probes due before sleep, sleep if scheduled, ask the probes due after sleep. After every
user has been run, the leakage harness probes across users.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from engram import __version__
from engram.bench.dataset import Benchmark, content_hash
from engram.bench.leakage import LeakCheck, run_leakage
from engram.bench.schema import Phase, Probe
from engram.bench.scoring import Score, format_table, score, summarise
from engram.systems.base import MemorySystem


@dataclass
class RunResult:
    scores: list[Score] = field(default_factory=list)
    leaks: list[LeakCheck] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


def ask(system: MemorySystem, question: str) -> str:
    """Probes run in their own session and must never be written to memory."""
    system.start_session(write=False)
    answer = system.respond(question)
    system.end_session()
    return answer


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run(
    bench: Benchmark,
    system: MemorySystem,
    *,
    leakage: bool = True,
    leak_targets_per_user: int = 3,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    started = time.time()
    sleeps = set(bench.sleeps)
    result = RunResult()
    clock = getattr(system, "set_clock", None)

    for user in bench.users:
        due: dict[tuple[int, Phase], list[Probe]] = defaultdict(list)
        for p in user.probes:
            due[(p.ask_after_session, p.phase)].append(p)

        def ask_due(s: int, phase: Phase, due=due) -> None:
            if clock:
                clock(s, phase.value)
            for p in due.get((s, phase), []):
                result.scores.append(score(p, ask(system, p.question)))

        system.load_user(user.id)
        for session in user.sessions:
            system.start_session(write=not session.incognito)
            for turn in session.turns:
                system.respond(turn.text)
            system.end_session()
            ask_due(session.index, Phase.PRE_SLEEP)
            if session.index in sleeps:
                system.sleep()
                ask_due(session.index, Phase.POST_SLEEP)
        system.unload_user()
        if progress:
            progress(f"{user.id}: {len(user.probes)} probes")

    if leakage:
        result.leaks = run_leakage(bench, system, targets_per_user=leak_targets_per_user)

    result.manifest = {
        "engram_version": __version__,
        "git_commit": _git_commit(),
        "system": system.describe(),
        "system_name": system.name,
        "bench": {"config": asdict(bench.config), "writer": bench.writer},
        "bench_sha256": content_hash(bench),
        "python": platform.python_version(),
        "host": platform.node(),
        "wall_clock_s": round(time.time() - started, 2),
        "n_scores": len(result.scores),
        "n_leak_checks": len(result.leaks),
    }
    return result


def save_run(result: RunResult, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
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


def apply_floor(scores: list[Score], floor: list[Score]) -> list[Score]:
    """Leak-verified floor (MERIT): drop answerable probes that the no-memory baseline already
    gets right, since a correct answer there cannot be evidence of memory."""
    known = {s.probe_id for s in floor if s.correct and s.category not in _NO_GOLD}
    return [s for s in scores if s.probe_id not in known]


_NO_GOLD = {"abstention", "forget", "incognito"}


def report(scores: list[Score], leaks: list[LeakCheck], title: str = "") -> str:
    parts = [format_table(summarise(scores), title or "Accuracy by category")]
    parts.append(format_table(summarise(scores, by="timing"), "\nAccuracy by probe timing"))
    if leaks:
        by_kind: dict[str, list[LeakCheck]] = defaultdict(list)
        for leak in leaks:
            by_kind[leak.kind].append(leak)
        parts.append("\nCross-user leakage (target: 0)")
        for kind, group in sorted(by_kind.items()):
            parts.append(f"{kind:<14} checks={len(group):>5} leaks={sum(g.leaked for g in group)}")
    return "\n".join(parts)
