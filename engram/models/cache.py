"""A persistent response cache, so interrupted runs resume and re-scoring is free.

Keyed by model name, messages and max_tokens. Only exact repeats hit the cache; a changed prompt is
a different key. Delete the cache file to force fresh calls.
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
    path: Path = field(default_factory=lambda: CACHE_ROOT / "responses.sqlite")
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, model TEXT, out TEXT)")
        self._db.commit()

    @property
    def name(self) -> str:
        return self.inner.name

    def _key(self, messages: list[Message], max_tokens: int) -> str:
        blob = json.dumps([self.inner.name, messages, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, messages: list[Message], *, max_tokens: int = 512) -> str:
        key = self._key(messages, max_tokens)
        with self._lock:
            row = self._db.execute("SELECT out FROM r WHERE k = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        out = self.inner.complete(messages, max_tokens=max_tokens)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO r VALUES (?, ?, ?)", (key, self.inner.name, out)
            )
            self._db.commit()
        self.misses += 1
        return out
