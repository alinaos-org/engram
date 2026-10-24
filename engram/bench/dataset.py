"""Generate, save and load an engram-bench dataset.

A dataset directory holds ``users.jsonl`` (one user per line: items, sessions, probes) and
``manifest.json`` (generator settings, seed, and a content hash), so any dataset can be rebuilt and
checked bit for bit. A sealed split's manifest withholds the seed and publishes a commitment to it
instead (``engram.bench.seal``); ``verify_seal`` rebuilds and checks it once the seed is revealed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from engram import __version__
from engram.bench import seal
from engram.bench.probes import make_probes, sleep_sessions
from engram.bench.schema import User, to_json, user_from_json
from engram.bench.world import fictional_name, fictional_surname, make_canary, plan_user
from engram.bench.writer import LongWriter, TemplateWriter
from engram.models.chat import NullModel


@dataclass
class BenchConfig:
    users: int = 50
    sessions: int = 30
    seed: int | None = 0  # None only in a sealed manifest, which withholds it
    sleep_every: int = 3
    late_sleeps: int = 3
    items_per_user: int = 10
    abstain_per_user: int = 3
    application_probes: bool = False  # v1: preference-application probes (changes the hash)


@dataclass
class Benchmark:
    config: BenchConfig
    users: list[User] = field(default_factory=list)
    writer: str = "template"
    sealed: dict | None = None  # the manifest's public commitment, for a sealed split
    usage: dict | None = None  # an LLM writer's model calls (cli.cmd_generate); not hashed

    @property
    def sleeps(self) -> list[int]:
        return sleep_sessions(self.config.sessions, self.config.sleep_every)

    def probes(self):
        for user in self.users:
            yield from user.probes


def generate(config: BenchConfig, writer: TemplateWriter | None = None) -> Benchmark:
    if config.seed is None:
        raise ValueError("config.seed is None: a sealed manifest withholds it (see verify_seal)")
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
        user.probes = make_probes(
            user, config.sessions, config.sleep_every, config.late_sleeps,
            application=config.application_probes,
        )  # fmt: skip
        users.append(user)
    return Benchmark(config, users, writer.name)


def _users_jsonl(bench: Benchmark) -> str:
    return "".join(json.dumps(to_json(u), ensure_ascii=False) + "\n" for u in bench.users)


def content_hash(bench: Benchmark) -> str:
    return hashlib.sha256(_users_jsonl(bench).encode()).hexdigest()


def _plan(user: User) -> str:
    """Everything about a user except the session text, which an LLM writer chose."""
    return json.dumps({k: v for k, v in to_json(user).items() if k != "sessions"})


def save(bench: Benchmark, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "users.jsonl").write_text(_users_jsonl(bench), encoding="utf-8")
    config = asdict(bench.config)
    if bench.sealed:
        config["seed"] = None
    manifest = {
        "engram_version": __version__,
        "config": config,
        "writer": bench.writer,
        "sha256": content_hash(bench),
        "n_users": len(bench.users),
        "n_probes": sum(len(u.probes) for u in bench.users),
    }
    if bench.usage:
        manifest["usage"] = bench.usage
    if bench.sealed:
        manifest["sealed"] = bench.sealed
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def load(path: Path, seed_file: Path | None = None) -> Benchmark:
    """Load and hash-check a dataset. A sealed one is checked against its commitment as well,
    and opened with ``seed_file`` if given; the seed never goes into ``bench.config``."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    try:
        users = [
            user_from_json(json.loads(line))
            # split on "\n" only: splitlines() also splits on U+2028, which JSON leaves raw
            for line in (path / "users.jsonl").read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: malformed users.jsonl ({e!r})") from e
    bench = Benchmark(
        BenchConfig(**manifest["config"]),
        users,
        manifest["writer"],
        sealed=manifest.get("sealed"),
        usage=manifest.get("usage"),
    )
    if content_hash(bench) != manifest["sha256"]:
        raise ValueError(f"{path}: users.jsonl does not match the manifest hash")
    if bench.sealed:
        seal.check_manifest(manifest)
        if seed_file is not None:
            seal.verify(bench.sealed, seal.read_seed_file(seed_file))
    elif seed_file is not None:
        raise ValueError(f"{path}: not a sealed dataset")
    return bench


def verify_seal(
    path: Path, seed_file: Path, report: Callable[[str], None] | None = None
) -> list[str]:
    """Check a revealed sealed split: the commitment, the content hash, and a regeneration.

    ``path`` is a dataset directory or a bare manifest file. Returns one line per passed check,
    each also given to ``report`` as it passes, and raises ``ValueError`` on the first failure;
    a caller that prints the lines so shows what passed before it. LLM-written splits are
    regenerated with writers that draw the same random numbers but call no model, so their plan
    (names, canaries, items, probes, gold) is checked but not the session text.
    """
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = manifest.get("sealed")
    if not sealed:
        raise ValueError(f"{path}: not a sealed dataset")
    seal.check_manifest(manifest)
    record = seal.read_seed_file(seed_file)
    seal.verify(sealed, record)
    lines: list[str] = []

    def ok(line: str) -> None:
        lines.append(line)
        if report is not None:
            report(line)

    ok("commitment: ok")
    published = None
    if path.is_dir() and (path / "users.jsonl").exists():
        published = load(path, seed_file)
        ok("content sha256: ok")

    def mismatch(what: str) -> ValueError:
        return ValueError(
            f"{path}: {what} differ from the regenerated split (generated at git commit "
            f"{sealed.get('git_commit')} with Python {sealed.get('python')}; regenerate with both)"
        )

    name = manifest["writer"]
    if name == "template" or name.startswith("llm:"):
        writer = TemplateWriter()
    elif name.startswith("long:"):
        writer = LongWriter(NullModel())
    else:
        raise ValueError(f"{path}: cannot regenerate writer {name!r}")
    bench = generate(BenchConfig(**{**manifest["config"], "seed": seal.seed_int(record)}), writer)
    n_probes = sum(len(u.probes) for u in bench.users)
    if (len(bench.users), n_probes) != (manifest["n_users"], manifest["n_probes"]):
        raise mismatch("counts")
    ok(f"regenerated counts: ok ({len(bench.users)} users, {n_probes} probes)")
    if name == "template":
        if content_hash(bench) != manifest["sha256"]:
            raise mismatch("contents")
        ok("regenerated content: ok")
    elif published is None:
        ok("regenerated plan: not checked (no users.jsonl)")
    else:
        for mine, theirs in zip(bench.users, published.users, strict=True):
            if _plan(mine) != _plan(theirs):
                raise mismatch(f"plans for {theirs.id}")
        ok("regenerated plan: ok")
    return lines
