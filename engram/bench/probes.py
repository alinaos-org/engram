"""Probe generation: questions with gold answers, asked at fixed points in the timeline.

Probe timing (eval-design.md):
  a  right after the session where the item last changed, before any sleep
  b  right after the next sleep
  c  right after the sleep ``late_sleeps`` sleeps later (durability and drift)
"""

from __future__ import annotations

from engram.bench.schema import Category, EventKind, ItemType, MemoryItem, Phase, Probe, User
from engram.bench.world import SLOT_BY_NAME

TYPE_CATEGORY = {
    ItemType.FACT: Category.FACT_RECALL,
    ItemType.PREFERENCE: Category.PREFERENCE,
    ItemType.DECISION: Category.DECISION,
    ItemType.CONSTRAINT: Category.PREFERENCE,
}


def sleep_sessions(n_sessions: int, sleep_every: int) -> list[int]:
    """Sessions after which a sleep (consolidation) runs."""
    return [s for s in range(n_sessions) if (s + 1) % sleep_every == 0]


def timing_points(
    session: int, sleeps: list[int], late_sleeps: int
) -> list[tuple[str, int, Phase]]:
    points = [("a", session, Phase.PRE_SLEEP)]
    after = [s for s in sleeps if s >= session]
    if after:
        points.append(("b", after[0], Phase.POST_SLEEP))
    if len(after) > late_sleeps:
        points.append(("c", after[late_sleeps], Phase.POST_SLEEP))
    return points


def _probe_for(
    item: MemoryItem, ask: int, phase: Phase, timing: str, pid: str, user_id: str
) -> Probe:
    slot = SLOT_BY_NAME[item.slot]
    value = item.value_at(ask)
    history = item.values_before(ask)
    if item.incognito_only:
        return Probe(
            pid, user_id, Category.INCOGNITO, slot.question, ask, phase, item.id,
            gold=None, reject=history, timing=timing,
        )  # fmt: skip
    if value is None:
        return Probe(
            pid, user_id, Category.FORGET, slot.question, ask, phase, item.id,
            gold=None, reject=history, timing=timing,
        )  # fmt: skip
    updated = any(c.event == EventKind.UPDATE and c.session <= ask for c in item.timeline)
    category = Category.UPDATE if updated else TYPE_CATEGORY[item.type]
    stale = [v for v in history if v != value]
    return Probe(
        pid, user_id, category, slot.question, ask, phase, item.id,
        gold=value, accept=[value], reject=stale, timing=timing,
    )  # fmt: skip


def make_probes(user: User, n_sessions: int, sleep_every: int, late_sleeps: int = 3) -> list[Probe]:
    sleeps = sleep_sessions(n_sessions, sleep_every)
    probes: list[Probe] = []
    seen: set[tuple[str, int, Phase]] = set()

    def add(item: MemoryItem, ask: int, phase: Phase, timing: str) -> None:
        key = (item.id, ask, phase)
        if key in seen:
            return
        seen.add(key)
        probes.append(_probe_for(item, ask, phase, timing, f"{user.id}-p{len(probes)}", user.id))

    for item in user.items:
        for change in item.timeline:
            for timing, ask, phase in timing_points(change.session, sleeps, late_sleeps):
                add(item, ask, phase, timing)

        # Temporal: what was the value before the latest update?
        updates = [c for c in item.timeline if c.event == EventKind.UPDATE]
        if updates:
            u = updates[-1]
            points = timing_points(u.session, sleeps, late_sleeps)
            timing, ask, phase = points[1] if len(points) > 1 else points[0]
            if item.value_at(ask) is not None:
                previous = item.values_before(u.session)[-2]
                slot = SLOT_BY_NAME[item.slot]
                probes.append(
                    Probe(
                        f"{user.id}-p{len(probes)}",
                        user.id,
                        Category.TEMPORAL,
                        f"Before it last changed, what was my {slot.label}?",
                        ask,
                        phase,
                        item.id,
                        gold=previous,
                        accept=[previous],
                        reject=[],  # naming the current value too is fine here
                        timing=timing,
                    )  # fmt: skip
                )

    # Abstention: slots the user never mentioned, asked mid-way and at the end.
    asked_slots = {i.slot for i in user.items}
    abstain_slots = [s for s in user.abstain_slots if s not in asked_slots]
    final = (n_sessions - 1, Phase.POST_SLEEP if (n_sessions - 1) in sleeps else Phase.PRE_SLEEP)
    middle = (n_sessions // 2, Phase.PRE_SLEEP)
    for slot_name in abstain_slots:
        slot = SLOT_BY_NAME[slot_name]
        for ask, phase in (middle, final):
            probes.append(
                Probe(
                    f"{user.id}-p{len(probes)}",
                    user.id,
                    Category.ABSTENTION,
                    slot.question,
                    ask,
                    phase,
                    None,
                    gold=None,
                    timing="a" if phase == Phase.PRE_SLEEP else "b",
                )  # fmt: skip
            )
    return probes
