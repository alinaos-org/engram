"""Data model for engram-bench: users, memory items, sessions, and probes.

Everything here is plain data and serialises to JSON. See research/eval-design.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ItemType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    CONSTRAINT = "constraint"


class EventKind(StrEnum):
    ASSERT = "assert"
    UPDATE = "update"
    FORGET_ITEM = "forget_item"
    FORGET_TOPIC = "forget_topic"
    FORGET_ALL = "forget_all"
    CANARY = "canary"


class Category(StrEnum):
    """Probe categories; numbers follow the task table in eval-design.md."""

    FACT_RECALL = "fact_recall"  # 1
    PREFERENCE = "preference"  # 2
    DECISION = "decision"  # 3
    UPDATE = "update"  # 4
    TEMPORAL = "temporal"  # 5
    ABSTENTION = "abstention"  # 6
    FORGET = "forget"  # 7
    INCOGNITO = "incognito"  # 8
    LEAKAGE = "leakage"  # 9 (run by the leakage harness)


class Phase(StrEnum):
    """When a probe is asked relative to the sleep that may follow its session."""

    PRE_SLEEP = "pre_sleep"
    POST_SLEEP = "post_sleep"


@dataclass
class Change:
    """One value in an item's timeline. ``value`` is None once the item is forgotten."""

    session: int
    value: str | None
    event: EventKind


@dataclass
class MemoryItem:
    id: str
    user_id: str
    slot: str
    type: ItemType
    topic: str
    timeline: list[Change] = field(default_factory=list)
    incognito_only: bool = False

    def value_at(self, session: int) -> str | None:
        """The item's value after ``session`` has ended, or None if unknown or forgotten."""
        value = None
        for change in self.timeline:
            if change.session <= session:
                value = change.value
        return value

    def values_before(self, session: int) -> list[str]:
        """Every non-null value the item held at or before ``session``."""
        return [c.value for c in self.timeline if c.session <= session and c.value is not None]

    def first_session(self) -> int:
        return self.timeline[0].session


@dataclass
class Event:
    kind: EventKind
    session: int
    item_id: str | None = None
    value: str | None = None
    previous: str | None = None
    topic: str | None = None


@dataclass
class Turn:
    """A scripted user message. The system under test writes the assistant turns."""

    text: str
    event: Event | None = None


@dataclass
class Session:
    user_id: str
    index: int
    day: int
    incognito: bool
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Probe:
    id: str
    user_id: str
    category: Category
    question: str
    ask_after_session: int
    phase: Phase
    item_id: str | None = None
    gold: str | None = None  # None means the correct behaviour is to abstain
    accept: list[str] = field(default_factory=list)
    reject: list[str] = field(default_factory=list)  # stale or forgotten values
    timing: str = "a"  # a = before sleep, b = after next sleep, c = after several sleeps


@dataclass
class User:
    id: str
    name: str
    canary: str
    items: list[MemoryItem] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    abstain_slots: list[str] = field(default_factory=list)  # never mentioned; for abstention

    def item(self, item_id: str) -> MemoryItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)


def to_json(obj: Any) -> Any:
    """Serialise dataclasses (with StrEnum members) into JSON-ready dicts."""
    return asdict(obj)


def user_from_json(d: dict[str, Any]) -> User:
    def event(e: dict | None) -> Event | None:
        if e is None:
            return None
        return Event(**{**e, "kind": EventKind(e["kind"])})

    items = [
        MemoryItem(
            **{
                **i,
                "type": ItemType(i["type"]),
                "timeline": [
                    Change(c["session"], c["value"], EventKind(c["event"])) for c in i["timeline"]
                ],
            }
        )
        for i in d["items"]
    ]
    sessions = [
        Session(
            **{**s, "turns": [Turn(text=t["text"], event=event(t["event"])) for t in s["turns"]]}
        )
        for s in d["sessions"]
    ]
    probes = [
        Probe(**{**p, "category": Category(p["category"]), "phase": Phase(p["phase"])})
        for p in d["probes"]
    ]
    return User(
        id=d["id"],
        name=d["name"],
        canary=d["canary"],
        items=items,
        sessions=sessions,
        probes=probes,
        abstain_slots=list(d.get("abstain_slots", [])),
    )
