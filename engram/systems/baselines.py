"""Baseline memory systems from research/eval-design.md.

B0 ``NoMemory``          base model, context wiped, nothing stored
B1 ``FullContext``       every earlier session pasted into the prompt (no wipe), up to a budget
B2/B3 ``RetrievalMemory`` retrieval over the user's session log; B2 uses a frontier model, B3
                          the same base model Engram will use. Same code, different ``ChatModel``.

Plus two controls that are not baselines:
``OracleMemory``  reads the gold world state: checks the scorers can reach 100%
``LeakyRetrieval`` one retrieval pool shared by all users: checks the leakage harness catches leaks

None of these baselines handle "forget" requests specially; they keep whatever they logged. That
is the honest behaviour of naive memory, and the forget category measures it.
"""

from __future__ import annotations

from collections import defaultdict

from engram.models.chat import ChatModel, Message, strip_think
from engram.systems.base import SYSTEM_PROMPT
from engram.systems.retrieval import BM25, Doc


class _Session:
    """Shared session bookkeeping: the current context and whether writes are allowed."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.user: str | None = None
        self.context: list[Message] = []
        self.write = True

    def load_user(self, user_id: str) -> None:
        self.user = user_id

    def unload_user(self) -> None:
        self.user = None

    def start_session(self, *, write: bool = True) -> None:
        self.context = []
        self.write = write

    def end_session(self) -> None:
        self.context = []

    def sleep(self) -> None:
        pass

    def _chat(self, system: str, message: str) -> str:
        messages = [{"role": "system", "content": system}, *self.context]
        messages.append({"role": "user", "content": message})
        reply = strip_think(self.model.complete(messages))
        self.context += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return reply


class NoMemory(_Session):
    """B0: the base model with nothing carried between sessions."""

    @property
    def name(self) -> str:
        return f"b0-nomemory:{self.model.name}"

    def respond(self, message: str) -> str:
        return self._chat(SYSTEM_PROMPT, message)

    def delete_user(self, user_id: str) -> None:
        pass

    def describe(self) -> dict:
        return {"system": "B0", "model": self.model.name}


class FullContext(_Session):
    """B1: all earlier (non-incognito) user messages are kept in the prompt.

    ``budget_chars`` approximates a context window; the oldest sessions drop out first, which is
    exactly the regime where memory should start to matter.
    """

    def __init__(self, model: ChatModel, budget_chars: int = 400_000) -> None:
        super().__init__(model)
        self.budget = budget_chars
        self.history: dict[str, list[str]] = defaultdict(list)

    @property
    def name(self) -> str:
        return f"b1-fullcontext:{self.model.name}"

    def respond(self, message: str) -> str:
        assert self.user is not None
        if self.write:
            self.history[self.user].append(message)
        past = self.history[self.user][:-1] if self.write else self.history[self.user]
        kept, size = [], 0
        for m in reversed(past):
            size += len(m)
            if size > self.budget:
                break
            kept.append(m)
        notes = "\n".join(f"- {m}" for m in reversed(kept))
        system = f"{SYSTEM_PROMPT}\n\nEverything the user has told you before:\n{notes}"
        return self._chat(system, message)

    def delete_user(self, user_id: str) -> None:
        self.history.pop(user_id, None)

    def describe(self) -> dict:
        return {"system": "B1", "model": self.model.name, "budget_chars": self.budget}


class RetrievalMemory(_Session):
    """B2/B3: BM25 retrieval over the user's own logged messages, gated by owner."""

    def __init__(self, model: ChatModel, k: int = 5, label: str = "b3") -> None:
        super().__init__(model)
        self.k = k
        self.label = label
        self.index = BM25()
        self.session_no = 0

    @property
    def name(self) -> str:
        return f"{self.label}-retrieval:{self.model.name}"

    def _owner_filter(self) -> str | None:
        return self.user

    def respond(self, message: str) -> str:
        assert self.user is not None
        hits = self.index.search(message, self.k, owner=self._owner_filter())
        notes = "\n".join(f"- {d.text}" for d in hits) or "- (nothing relevant found)"
        system = f"{SYSTEM_PROMPT}\n\nPossibly relevant things the user told you before:\n{notes}"
        reply = self._chat(system, message)
        if self.write:
            self.index.add(Doc(message, self.user, {"session": self.session_no}))
        return reply

    def end_session(self) -> None:
        super().end_session()
        self.session_no += 1

    def delete_user(self, user_id: str) -> None:
        self.index.remove(lambda d: d.owner == user_id)

    def describe(self) -> dict:
        return {"system": self.label.upper(), "model": self.model.name, "k": self.k, "bm25": True}


class LeakyRetrieval(RetrievalMemory):
    """Control only: retrieval without ownership gating, as in MemLeak's pooled setting."""

    def __init__(self, model: ChatModel, k: int = 5) -> None:
        super().__init__(model, k, label="leaky")

    def _owner_filter(self) -> str | None:
        return None


class OracleMemory:
    """Control only: answers probes from the gold world state. Should score ~100%."""

    name = "oracle"

    def __init__(self, bench) -> None:
        self.answers: dict[tuple[str, str, int, str], str] = {}
        self.bench = bench
        self.user: str | None = None
        self.clock: tuple[int, str] = (0, "pre_sleep")

    def load_user(self, user_id: str) -> None:
        self.user = user_id

    def unload_user(self) -> None:
        self.user = None

    def start_session(self, *, write: bool = True) -> None:
        pass

    def set_clock(self, session: int, phase: str) -> None:
        """The runner tells the oracle where it is; real systems never get this."""
        self.clock = (session, phase)

    def respond(self, message: str) -> str:
        for user in self.bench.users:
            if user.id != self.user:
                continue
            for p in user.probes:
                if p.question == message and (p.ask_after_session, p.phase.value) == self.clock:
                    return p.gold if p.gold is not None else "I don't know."
        return "OK."

    def end_session(self) -> None:
        pass

    def sleep(self) -> None:
        pass

    def delete_user(self, user_id: str) -> None:
        pass

    def describe(self) -> dict:
        return {"system": "oracle"}
