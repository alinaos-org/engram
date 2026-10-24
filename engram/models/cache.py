"""A persistent response cache, so interrupted runs resume and re-scoring is free.

Keyed by model name, messages and max_tokens. Only exact repeats hit the cache; a changed prompt is
a different key. Delete the cache file to force fresh calls. With ``path=None`` (``--no-cache``)
nothing is stored, but calls are still counted: ``usage`` reports them for run manifests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from engram.models.chat import ChatModel, Message
from engram.models.subscription import CACHE_ROOT


@dataclass
class CachedModel:
    inner: ChatModel
    path: Path | None = field(default_factory=lambda: CACHE_ROOT / "responses.sqlite")
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, model TEXT, out TEXT)")
        self._db.commit()

    @property
    def name(self) -> str:
        return self.inner.name

    def _key(self, messages: list[Message], max_tokens: int) -> str:
        tag = getattr(self.inner, "cache_tag", "")
        blob = json.dumps([self.inner.name, messages, max_tokens], sort_keys=True)
        if tag:  # config-sensitive models (CLI isolation settings) key on their config too
            blob += "|" + tag
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        key = self._key(messages, max_tokens)
        with self._lock:  # counters are updated under the lock too: runs call from many threads
            if self._db is not None:
                row = self._db.execute("SELECT out FROM r WHERE k = ?", (key,)).fetchone()
                if row is not None:
                    self.hits += 1
                    return row[0]
            self.misses += 1  # counted before the call, so a call that raises is counted too
        out = self.inner.complete(messages, max_tokens=max_tokens)
        if self._db is not None:
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO r VALUES (?, ?, ?)", (key, self.inner.name, out)
                )
                self._db.commit()
        return out


def usage(model: ChatModel) -> dict:
    """What one model role drew, for manifests: its spec, the calls that reached the backend
    (cache misses, failed ones included) and the cache hits. A ``claude:``/``codex:`` spec adds
    its CLI invocations (``attempts``, retries included) and the shared subscription pool they
    draw on. A model outside ``CachedModel`` (a mock) is not counted."""
    out: dict = {"model": model.name}
    if isinstance(model, CachedModel):
        out |= {"calls": model.misses, "cache_hits": model.hits}
    inner = getattr(model, "inner", model)
    if (attempts := getattr(inner, "attempts", None)) is not None:
        out["attempts"] = attempts
    if pool := getattr(inner, "pool", None):
        out["pool"] = pool
    return out
