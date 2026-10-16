"""Synthetic users and their memory timelines.

Every user is fictitious and every planted value is invented (or chosen at random), so the base
model cannot already know it. This follows TOFU's fictitious-profile design.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from engram.bench.schema import Change, Event, EventKind, ItemType, MemoryItem

_SYL_A = ["ka", "lo", "ve", "mi", "tor", "sa", "ren", "ul", "dra", "fe", "qui", "bo", "zel", "ny"]
_SYL_B = ["ra", "vin", "da", "nel", "mor", "is", "ten", "ka", "lio", "brec", "set", "ane", "o"]
_SURNAME_END = ["ville", "ström", "ade", "ovich", "worth", "enne", "aro", "ski", "ley", "doux"]


def fictional_name(rng: random.Random, words: int = 1) -> str:
    parts = []
    for _ in range(words):
        parts.append((rng.choice(_SYL_A) + rng.choice(_SYL_B)).capitalize())
    return " ".join(parts)


def fictional_surname(rng: random.Random) -> str:
    return (rng.choice(_SYL_A) + rng.choice(_SYL_B) + rng.choice(_SURNAME_END)).capitalize()


def _pick(options: list[str]) -> Callable[[random.Random], str]:
    return lambda rng: rng.choice(options)


def _company(rng: random.Random) -> str:
    return f"{fictional_name(rng)} {rng.choice(['Labs', 'Systems', 'Works', 'Analytics'])}"


def _restaurant(rng: random.Random) -> str:
    return f"{rng.choice(['Café', 'Trattoria', 'Bistro', 'Kitchen'])} {fictional_name(rng)}"


def _codename(rng: random.Random) -> str:
    return f"Project {fictional_name(rng)}"


@dataclass(frozen=True)
class Slot:
    name: str
    label: str  # how the user refers to it, e.g. "sister's name"
    type: ItemType
    topic: str
    values: Callable[[random.Random], str]
    assert_t: tuple[str, ...]
    update_t: tuple[str, ...]
    question: str
    mutable: bool = True


SLOTS: list[Slot] = [
    Slot(
        "sister_name",
        "sister's name",
        ItemType.FACT,
        "family",
        lambda r: fictional_name(r),
        ("My sister {v} is visiting next month.", "I was on the phone with my sister {v}."),
        ("Small correction: my sister goes by {v} now, not {old}.",),
        "What is my sister's name?",
        mutable=False,
    ),
    Slot(
        "partner_name",
        "partner's name",
        ItemType.FACT,
        "family",
        lambda r: fictional_name(r),
        ("My partner {v} thinks I work too late.",),
        ("{old} and I split up; I'm seeing someone called {v} now.",),
        "What is my partner's name?",
    ),
    Slot(
        "city",
        "city",
        ItemType.FACT,
        "home",
        _pick(["Lyon", "Tromsø", "Porto", "Kraków", "Valparaíso", "Hobart", "Ljubljana", "Ghent"]),
        ("I live in {v}, so time zones are always a pain.",),
        ("Big news: I moved from {old} to {v} last week.",),
        "Which city do I live in?",
    ),
    Slot(
        "pet_name",
        "dog's name",
        ItemType.FACT,
        "home",
        lambda r: fictional_name(r),
        ("My dog {v} keeps stealing socks.",),
        ("We renamed the dog from {old} to {v}, long story.",),
        "What is my dog's name?",
        mutable=False,
    ),
    Slot(
        "employer",
        "employer",
        ItemType.FACT,
        "work",
        _company,
        ("I work at {v} on the data platform team.",),
        ("I left {old}; I started at {v} on Monday.",),
        "Where do I work?",
    ),
    Slot(
        "manager_name",
        "manager's name",
        ItemType.FACT,
        "work",
        lambda r: f"{fictional_name(r)} {fictional_surname(r)}",
        ("My manager {v} wants the roadmap by Friday.",),
        ("Reorg: my new manager is {v}, {old} moved teams.",),
        "What is my manager's name?",
    ),
    Slot(
        "allergy",
        "allergy",
        ItemType.FACT,
        "health",
        _pick(["peanuts", "shellfish", "sesame", "penicillin", "latex", "kiwi"]),
        ("Heads up for recipes: I'm allergic to {v}.",),
        ("Turns out the allergy test was wrong: it's {v}, not {old}.",),
        "What am I allergic to?",
    ),
    Slot(
        "medication",
        "medication",
        ItemType.FACT,
        "health",
        lambda r: f"{fictional_name(r).lower()}ex",
        ("I take {v} every morning, so no grapefruit for me.",),
        ("My doctor switched me from {old} to {v}.",),
        "Which medication do I take every morning?",
    ),
    Slot(
        "restaurant",
        "favourite restaurant",
        ItemType.PREFERENCE,
        "food",
        _restaurant,
        ("{v} is my favourite place to eat, by far.",),
        ("{old} went downhill; {v} is my favourite now.",),
        "What is my favourite restaurant?",
    ),
    Slot(
        "code_language",
        "preferred language for code examples",
        ItemType.PREFERENCE,
        "tech",
        _pick(["Rust", "Go", "Kotlin", "Elixir", "Zig", "OCaml", "Haskell", "Julia"]),
        ("When you show me code, please use {v}.", "I'd rather see examples in {v}."),
        ("I've switched teams, so show me {v} examples now instead of {old}.",),
        "Which programming language do I want code examples in?",
    ),
    Slot(
        "editor",
        "editor",
        ItemType.PREFERENCE,
        "tech",
        _pick(["Neovim", "Helix", "Emacs", "Zed", "Kakoune", "Sublime Text"]),
        ("I do everything in {v}, so keybinding tips should assume that.",),
        ("I finally moved from {old} to {v}.",),
        "Which editor do I use?",
    ),
    Slot(
        "units",
        "units",
        ItemType.PREFERENCE,
        "tech",
        _pick(["metric", "imperial"]),
        ("Always give me measurements in {v} units.",),
        ("Please use {v} units from now on, not {old}.",),
        "Which units do I want measurements in?",
    ),
    Slot(
        "database",
        "database decision",
        ItemType.DECISION,
        "projects",
        _pick(["self-hosted Postgres", "SQLite", "DuckDB", "FoundationDB", "CockroachDB"]),
        ("We decided to go with {v} for the new service.",),
        ("We reversed the database decision: {v} instead of {old}, for cost reasons.",),
        "Which database did we decide to use for the new service?",
    ),
    Slot(
        "project_codename",
        "project codename",
        ItemType.DECISION,
        "projects",
        _codename,
        ("The migration is officially called {v}.",),
        ("Legal made us rename {old}; it's {v} now.",),
        "What is the codename of the migration project?",
    ),
    Slot(
        "release_day",
        "release day",
        ItemType.DECISION,
        "projects",
        _pick(["Monday", "Tuesday", "Wednesday", "Thursday"]),
        ("We agreed to ship releases every {v}.",),
        ("Release day moves from {old} to {v}.",),
        "Which day of the week did we agree to ship releases?",
    ),
    Slot(
        "diet",
        "diet",
        ItemType.CONSTRAINT,
        "food",
        _pick(["vegetarian", "pescatarian", "vegan", "gluten-free", "halal", "kosher"]),
        ("For any food suggestions: I'm {v}.",),
        ("I'm not {old} any more; I'm {v} now.",),
        "What diet do I follow?",
    ),
    Slot(
        "meeting_window",
        "meeting window",
        ItemType.CONSTRAINT,
        "schedule",
        _pick(["after 10am", "before 3pm", "on Tuesday or Thursday", "after 2pm"]),
        ("Only book meetings for me {v}.",),
        ("New rule: meetings only {v}, not {old}.",),
        "When am I willing to have meetings?",
    ),
    Slot(
        "gym_day",
        "climbing day",
        ItemType.CONSTRAINT,
        "schedule",
        _pick(["Mondays", "Wednesdays", "Fridays", "Saturdays"]),
        ("I go climbing on {v}, so keep those evenings free.",),
        ("Climbing moved from {old} to {v}.",),
        "Which day do I go climbing?",
    ),
]

SLOT_BY_NAME = {s.name: s for s in SLOTS}

_CANARY_WORDS = ["ALDER", "QUILL", "MOSS", "EMBER", "TIDE", "FLINT", "OPAL", "WREN", "BRINE"]


def make_canary(rng: random.Random) -> str:
    return f"{rng.randint(1, 9)}-{rng.choice(_CANARY_WORDS)}-{rng.randint(1000, 9999)}"


@dataclass
class Plan:
    """The scheduled world for one user before it is rendered into sessions."""

    items: list[MemoryItem]
    events: dict[int, list[Event]]
    incognito_sessions: set[int]
    abstain_slots: list[str]
    days: list[int]


def plan_user(
    rng: random.Random,
    user_id: str,
    n_sessions: int,
    canary: str,
    *,
    n_items: int = 10,
    n_abstain: int = 3,
    p_update: float = 0.4,
    p_forget_topic: float = 0.5,
    p_forget_all: float = 0.1,
) -> Plan:
    """Schedule asserts, updates, forget requests, incognito sessions and the canary."""
    if n_sessions < 6:
        raise ValueError("need at least 6 sessions per user")
    slots = rng.sample(SLOTS, len(SLOTS))
    n_incognito = 1 if n_sessions < 12 else 2
    asserted = slots[:n_items]
    incognito_slots = slots[n_items : n_items + n_incognito]
    abstain = [s.name for s in slots[n_items + n_incognito : n_items + n_incognito + n_abstain]]

    events: dict[int, list[Event]] = {s: [] for s in range(n_sessions)}
    items: list[MemoryItem] = []
    events[0].append(Event(EventKind.CANARY, 0, value=canary))

    last_assert = max(1, int(n_sessions * 0.6))
    incognito_sessions = set(rng.sample(range(1, n_sessions - 1), n_incognito))
    normal_sessions = [s for s in range(last_assert) if s not in incognito_sessions]

    for k, slot in enumerate(asserted):
        item = MemoryItem(f"{user_id}-i{k}", user_id, slot.name, slot.type, slot.topic)
        s = rng.choice(normal_sessions)
        v = slot.values(rng)
        item.timeline.append(Change(s, v, EventKind.ASSERT))
        events[s].append(Event(EventKind.ASSERT, s, item.id, v))
        later = [x for x in range(s + 2, n_sessions - 1) if x not in incognito_sessions]
        if slot.mutable and later and rng.random() < p_update:
            u = rng.choice(later)
            nv = slot.values(rng)
            while nv == v:
                nv = slot.values(rng)
            item.timeline.append(Change(u, nv, EventKind.UPDATE))
            events[u].append(Event(EventKind.UPDATE, u, item.id, nv, previous=v))
        items.append(item)

    for k, (slot, s) in enumerate(zip(incognito_slots, sorted(incognito_sessions), strict=True)):
        item = MemoryItem(
            f"{user_id}-x{k}", user_id, slot.name, slot.type, slot.topic, incognito_only=True
        )
        v = slot.values(rng)
        item.timeline.append(Change(s, v, EventKind.ASSERT))
        events[s].append(Event(EventKind.ASSERT, s, item.id, v))
        items.append(item)

    def last_event_session(item: MemoryItem) -> int:
        return item.timeline[-1].session

    def schedule_forget(targets: list[MemoryItem], kind: EventKind, topic: str | None) -> bool:
        earliest = max(last_event_session(i) for i in targets) + 1
        choices = [x for x in range(earliest, n_sessions - 1) if x not in incognito_sessions]
        if not choices:
            return False
        f = rng.choice(choices)
        for i in targets:
            i.timeline.append(Change(f, None, kind))
        item_id = targets[0].id if kind == EventKind.FORGET_ITEM else None
        events[f].append(Event(kind, f, item_id=item_id, topic=topic))
        return True

    remembered = [i for i in items if not i.incognito_only]
    forgotten: set[str] = set()
    for item in rng.sample(remembered, len(remembered)):
        if schedule_forget([item], EventKind.FORGET_ITEM, None):
            forgotten.add(item.id)
            break

    if rng.random() < p_forget_topic:
        topics = sorted({i.topic for i in remembered})
        rng.shuffle(topics)
        for topic in topics:
            group = [i for i in remembered if i.topic == topic and i.id not in forgotten]
            if group and len(group) == len([i for i in remembered if i.topic == topic]):
                if schedule_forget(group, EventKind.FORGET_TOPIC, topic):
                    forgotten.update(i.id for i in group)
                break

    if rng.random() < p_forget_all:
        live = [i for i in remembered if i.id not in forgotten]
        if live:
            schedule_forget(live, EventKind.FORGET_ALL, None)

    days, day = [], 0
    for _ in range(n_sessions):
        days.append(day)
        day += rng.randint(1, 5)
    return Plan(items, events, incognito_sessions, abstain, days)
