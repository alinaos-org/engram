"""Generate, save and load an engram-bench dataset.

A dataset directory holds ``users.jsonl`` (one user per line: items, sessions, probes) and
``manifest.json`` (generator settings, seed, and a content hash), so any dataset can be rebuilt and
checked bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from engram import __version__
from engram.bench.probes import make_probes, sleep_sessions
from engram.bench.schema import User, to_json, user_from_json
from engram.bench.world import fictional_name, fictional_surname, make_canary, plan_user
from engram.bench.writer import TemplateWriter


@dataclass
class BenchConfig:
    users: int = 50
    sessions: int = 30
    seed: int = 0
    sleep_every: int = 3
    late_sleeps: int = 3
    items_per_user: int = 10
    abstain_per_user: int = 3


@dataclass
class Benchmark:
    config: BenchConfig
    users: list[User] = field(default_factory=list)
    writer: str = "template"

    @property
    def sleeps(self) -> list[int]:
        return sleep_sessions(self.config.sessions, self.config.sleep_every)

    def probes(self):
        for user in self.users:
            yield from user.probes


def generate(config: BenchConfig, writer: TemplateWriter | None = None) -> Benchmark:
    writer = writer or TemplateWriter()
    rng = random.Random(config.seed)
    canaries: set[str] = set()
    users = []
    for n in range(config.users):
        uid = f"u{n:04d}"
        canary = make_canary(rng)
        while canary in canaries:
            canary = make_canary(rng)
        canaries.add(canary)
        plan = plan_user(
            rng,
            uid,
            config.sessions,
            canary,
            n_items=config.items_per_user,
            n_abstain=config.abstain_per_user,
        )
        user = User(
            uid,
            f"{fictional_name(rng)} {fictional_surname(rng)}",
            canary,
            items=plan.items,
            abstain_slots=plan.abstain_slots,
        )
        user.sessions = writer.write(user, plan, rng)
        user.probes = make_probes(user, config.sessions, config.sleep_every, config.late_sleeps)
        users.append(user)
    return Benchmark(config, users, writer.name)


def _users_jsonl(bench: Benchmark) -> str:
    return "".join(json.dumps(to_json(u), ensure_ascii=False) + "\n" for u in bench.users)


def content_hash(bench: Benchmark) -> str:
    return hashlib.sha256(_users_jsonl(bench).encode()).hexdigest()


def save(bench: Benchmark, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "users.jsonl").write_text(_users_jsonl(bench), encoding="utf-8")
    manifest = {
        "engram_version": __version__,
        "config": asdict(bench.config),
        "writer": bench.writer,
        "sha256": content_hash(bench),
        "n_users": len(bench.users),
        "n_probes": sum(len(u.probes) for u in bench.users),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def load(path: Path) -> Benchmark:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    try:
        users = [
            user_from_json(json.loads(line))
            for line in (path / "users.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: malformed users.jsonl ({e!r})") from e
    bench = Benchmark(BenchConfig(**manifest["config"]), users, manifest["writer"])
    if content_hash(bench) != manifest["sha256"]:
        raise ValueError(f"{path}: users.jsonl does not match the manifest hash")
    return bench
