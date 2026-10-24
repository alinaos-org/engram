"""Sealed held-out splits: a private seed, a public commitment to it, and a work-tree guard.

The scheme is in research/sealed-split.md. A seed file holds the seed, a salt and the commitment
``sha256("engram-seal-v1|<salt>|<seed>|<spec_sha256>")``. The dataset manifest publishes only the
commitment, so the split can be checked after the seed is revealed. This module imports nothing
from engram, so a verifier can read the whole format here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path

SCHEME = "engram-seal-v1"
PER_PROBE = "per-probe content of a sealed split"
_FIELDS = ("scheme", "seed", "salt", "spec_sha256", "seed_commitment")
_HEX = re.compile(r"[0-9a-f]{64}")


def spec_sha256(config: dict, writer: str) -> str:
    """Hash of the manifest's config (as stored, minus ``seed``) and the writer name."""
    spec = {"config": {k: v for k, v in config.items() if k != "seed"}, "writer": writer}
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def commitment(salt: str, seed: str, spec: str) -> str:
    # Each field is exactly 64 hex characters, so the joined string is unambiguous.
    return hashlib.sha256(f"{SCHEME}|{salt}|{seed}|{spec}".encode("ascii")).hexdigest()


def new_record(spec: str) -> dict:
    """A fresh seed-file record: 256-bit seed and salt from the OS CSPRNG."""
    seed, salt = secrets.token_hex(32), secrets.token_hex(32)
    return {
        "scheme": SCHEME,
        "seed": seed,
        "salt": salt,
        "spec_sha256": spec,
        "seed_commitment": commitment(salt, seed, spec),
    }


def seed_int(record: dict) -> int:
    return int(record["seed"], 16)


def sealed_block(record: dict) -> dict:
    """The public part of a record, for the manifest's ``sealed`` object."""
    return {k: record[k] for k in ("scheme", "spec_sha256", "seed_commitment")}


def write_seed_file(path: Path, record: dict, *, dataset_dir: Path) -> None:
    """Write ``record`` once with mode 0600. There is no override for any refusal."""
    root = work_tree(path)
    if root is not None:
        raise ValueError(
            f"{path} is inside the git work tree {root}; "
            "keep the seed file outside every repository"
        )
    if path.resolve().is_relative_to(dataset_dir.resolve()):
        raise ValueError(
            f"{path} is inside the dataset directory {dataset_dir}, which is copied to run hosts; "
            "keep the seed file outside it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(f"{path} exists; a seed file is never overwritten") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2) + "\n")


def read_seed_file(path: Path) -> dict:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError:
        record = None
    if not (
        isinstance(record, dict)
        and set(record) == set(_FIELDS)
        and record["scheme"] == SCHEME
        and all(isinstance(record[k], str) and _HEX.fullmatch(record[k]) for k in _FIELDS[1:])
    ):
        raise ValueError(f"{path}: not an engram seed file")
    return record


def check_manifest(manifest: dict) -> None:
    """A sealed manifest withholds its seed, and its spec hash matches its config and writer."""
    sealed = manifest["sealed"]
    if manifest["config"].get("seed") is not None:
        raise ValueError("sealed manifest has a seed; a sealed split never publishes it")
    if sealed.get("scheme") != SCHEME:
        raise ValueError(f"sealed manifest has an unknown scheme {sealed.get('scheme')!r}")
    if spec_sha256(manifest["config"], manifest["writer"]) != sealed.get("spec_sha256"):
        raise ValueError("sealed manifest's config and writer do not match its spec_sha256")


def verify(sealed: dict, record: dict) -> None:
    """Check that a seed-file record opens the manifest's commitment."""
    if (
        record["spec_sha256"] != sealed["spec_sha256"]
        or record["seed_commitment"] != sealed["seed_commitment"]
    ):
        raise ValueError("seed file belongs to a different split")
    opened = commitment(record["salt"], record["seed"], sealed["spec_sha256"])
    if opened != sealed["seed_commitment"]:
        raise ValueError("seed and salt do not open the commitment")


def work_tree(path: Path) -> Path | None:
    """The git work tree holding ``path``, if any: the first of its resolved path and parents
    with a ``.git`` entry (a directory, or the file worktrees and submodules use). Resolving
    first catches symlinks into a repository; no git binary is needed. ``~`` is not expanded:
    the path is checked where it will be written (the CLI expands ``~`` before both)."""
    resolved = path.resolve()
    for d in (resolved, *resolved.parents):
        if os.path.lexists(d / ".git"):
            return d
    return None


def refuse_in_repo(path: Path, what: str, *, allow: bool = False) -> None:
    root = None if allow else work_tree(path)
    if root is not None:
        raise ValueError(
            f"{path} is inside the git work tree {root}; {what} stays outside it "
            "(pass --allow-in-repo to override)"
        )
