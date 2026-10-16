"""JSONL and config helpers shared by the corpus stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    # split on "\n" only: str.splitlines() also splits on U+2028, which JSON leaves raw
    with open(path, encoding="utf-8", newline="\n") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: bad JSON ({e})") from e


def read_many(paths: Iterable[Path | str]) -> Iterator[dict[str, Any]]:
    for p in paths:
        yield from read_jsonl(p)


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a JSON config, or YAML when PyYAML is installed and the suffix says so."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise SystemExit(f"{path}: YAML configs need PyYAML; use the JSON config") from e
        return yaml.safe_load(text)
    return json.loads(text)
