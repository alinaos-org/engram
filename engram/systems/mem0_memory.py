"""B5: Mem0 (open source) as an extract-and-consolidate text-memory baseline.

Mem0 extracts facts from each finished session with an LLM and decides whether to add, update or
delete stored memories. That is the text-space analogue of Engram's sleep, and the fair comparison
for the update and forget categories, where naive retrieval failed in the pilot.

Mem0's own LLM calls go through a local OpenAI-compatible shim (engram/models/serve.py) in front
of the same ``ChatModel`` used for answering, so subscription models and the response cache work
unchanged. Embeddings run locally (bge-small); the vector store is a local, in-memory Qdrant.
Telemetry is switched off before Mem0 is imported. Needs ``mem0ai`` (the ``baselines`` extra).

Mem0 keeps one store for every user, scoped by ``user_id``, so this baseline runs with workers=1.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from engram.models.chat import ChatModel, strip_think
from engram.models.serve import BackgroundServer
from engram.systems.base import SYSTEM_PROMPT

EMBEDDER = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384


class Mem0Memory:
    def __init__(
        self,
        model: ChatModel,
        *,
        k: int = 5,
        infer: bool = True,
        workdir: Path | None = None,
        embedder: str = EMBEDDER,
    ) -> None:
        os.environ["MEM0_TELEMETRY"] = "False"  # must be set before mem0 is imported
        from mem0 import Memory

        self.model = model
        self.k = k
        self.infer = infer
        self._tmp = tempfile.TemporaryDirectory(prefix="engram-mem0-") if workdir is None else None
        self.workdir = Path(self._tmp.name if self._tmp else workdir)
        self.server = BackgroundServer(model).__enter__()
        self.mem = Memory.from_config(
            {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": model.name,
                        "openai_base_url": self.server.base_url,
                        "api_key": "engram-local",
                        "temperature": 0.0,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": embedder, "embedding_dims": EMBED_DIMS},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "engram",
                        "path": str(self.workdir / "qdrant"),
                        "embedding_model_dims": EMBED_DIMS,
                        "on_disk": False,
                    },
                },
                "history_db_path": str(self.workdir / "history.db"),
            }
        )
        self.user: str | None = None
        self.context: list[dict] = []
        self.buffer: list[str] = []
        self.write = True
        self.adds = 0

    @property
    def name(self) -> str:
        return f"b5-mem0:{self.model.name}"

    def load_user(self, user_id: str) -> None:
        self.user = user_id

    def unload_user(self) -> None:
        self.user = None

    def start_session(self, *, write: bool = True, day: int | None = None) -> None:
        self.context, self.buffer, self.write = [], [], write

    def observe(self, message: str) -> None:
        self.context.append({"role": "user", "content": message})
        if self.write:
            self.buffer.append(message)

    def respond(self, message: str) -> str:
        assert self.user is not None
        found = self.mem.search(message, top_k=self.k, filters={"user_id": self.user})
        notes = [r.get("memory", "") for r in (found or {}).get("results", [])]
        listing = "\n".join(f"- {n}" for n in notes if n) or "- (nothing relevant found)"
        # Mem0 dates what it extracts by the real clock, so the answering model must share that
        # clock; otherwise a memory written "today" in UTC can look like it is in the future.
        today = datetime.now(UTC).date().isoformat()
        system = (
            f"{SYSTEM_PROMPT}\nToday's date: {today}.\n\n"
            f"What you remember about the user:\n{listing}"
        )
        messages = [{"role": "system", "content": system}, *self.context]
        messages.append({"role": "user", "content": message})
        reply = strip_think(self.model.complete(messages))
        self.observe(message)
        self.context.append({"role": "assistant", "content": reply})
        return reply

    def end_session(self) -> None:
        """Hand the finished session to Mem0, which extracts, updates and deletes memories."""
        if self.write and self.buffer and self.user is not None:
            self.mem.add(
                [{"role": "user", "content": m} for m in self.buffer],
                user_id=self.user,
                infer=self.infer,
            )
            self.adds += 1
        self.context, self.buffer = [], []

    def sleep(self) -> None:
        pass  # Mem0 consolidates on every add

    def delete_user(self, user_id: str) -> None:
        self.mem.delete_all(user_id=user_id)

    def describe(self) -> dict:
        return {"system": "B5", "model": self.model.name, "k": self.k, "infer": self.infer,
                "embedder": EMBEDDER, "mem0": _mem0_version()}  # fmt: skip

    def close(self) -> None:
        self.server.__exit__(None, None, None)
        if self._tmp:
            self._tmp.cleanup()


def _mem0_version() -> str:
    try:
        from importlib.metadata import version

        return version("mem0ai")
    except Exception:
        return "unknown"
