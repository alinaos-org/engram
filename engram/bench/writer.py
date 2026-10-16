"""Render a user's planned events into scripted sessions.

``TemplateWriter`` is deterministic and needs no model; CI and unit tests use it. ``LLMWriter``
rewrites each planted message into natural, varied text with a chat model, and falls back to the
template whenever the rewrite drops a value the probes depend on.
"""

from __future__ import annotations

import random

from engram.bench.schema import Event, EventKind, Session, Turn, User
from engram.bench.world import SLOT_BY_NAME, Plan
from engram.models.chat import ChatModel, strip_think

TOPIC_PHRASE = {
    "family": "my family",
    "home": "my home life",
    "work": "my work",
    "health": "my health",
    "food": "food and what I eat",
    "tech": "my tech setup",
    "projects": "our projects",
    "schedule": "my schedule",
}

FILLER = [
    "Can you suggest a name for a git branch that refactors our logging?",
    "What's a good way to explain backpressure to a new engineer?",
    "Give me three ideas for a team offsite activity.",
    "How should I structure a one-page design doc?",
    "What's the difference between a mutex and a semaphore, briefly?",
    "Draft a polite reply declining a meeting invite.",
    "What are some good questions to ask in a 1:1?",
    "Summarise the trade-offs of monorepos in a few bullets.",
    "How do I keep a long-running migration safe to roll back?",
    "Suggest a weekend reading list on distributed systems.",
    "What's a sensible on-call handover checklist?",
    "Help me word a status update that says we're a week late.",
    "How do I write a good commit message?",
    "Explain idempotency keys like I'm a product manager.",
    "What should go in a postmortem template?",
]


def template_text(event: Event, user: User, rng: random.Random) -> str:
    if event.kind == EventKind.CANARY:
        return f"Note this down for me: my locker code is {event.value}."
    if event.kind == EventKind.FORGET_TOPIC:
        return f"Please forget everything I've told you about {TOPIC_PHRASE[event.topic or '']}."
    if event.kind == EventKind.FORGET_ALL:
        return "Please reset and forget everything you know about me."
    item = user.item(event.item_id or "")
    slot = SLOT_BY_NAME[item.slot]
    if event.kind == EventKind.FORGET_ITEM:
        return f"Please forget my {slot.label}. I don't want you to keep that."
    if event.kind == EventKind.ASSERT:
        return rng.choice(slot.assert_t).format(v=event.value)
    if event.kind == EventKind.UPDATE:
        return rng.choice(slot.update_t).format(v=event.value, old=event.previous)
    raise ValueError(event.kind)


def required_strings(event: Event) -> list[str]:
    """Strings a rewrite must keep verbatim for the gold answers to stay valid."""
    out = [event.value] if event.value else []
    if event.kind == EventKind.UPDATE and event.previous:
        out.append(event.previous)
    return out


class TemplateWriter:
    name = "template"

    def __init__(self, filler_per_session: tuple[int, int] = (1, 3)) -> None:
        self.filler = filler_per_session

    def rewrite(self, text: str, event: Event, rng: random.Random) -> str:
        return text

    def write(self, user: User, plan: Plan, rng: random.Random) -> list[Session]:
        sessions = []
        for s, day in enumerate(plan.days):
            turns = [Turn(rng.choice(FILLER))]
            for event in plan.events[s]:
                text = self.rewrite(template_text(event, user, rng), event, rng)
                turns.append(Turn(text, event))
            for _ in range(rng.randint(*self.filler) - 1):
                turns.append(Turn(rng.choice(FILLER)))
            sessions.append(
                Session(user.id, s, day, incognito=s in plan.incognito_sessions, turns=turns)
            )
        return sessions


class LLMWriter(TemplateWriter):
    """Paraphrases planted messages with a chat model so they read like real conversation."""

    PROMPT = (
        "Rewrite the message below as something a busy professional might type to an AI "
        "assistant: casual, one to three sentences, possibly mentioning it in passing while "
        "asking for something else. You MUST keep these exact strings verbatim: {keep}. "
        "Reply with the rewritten message only.\n\nMessage: {text}"
    )

    def __init__(self, model: ChatModel, **kw) -> None:
        super().__init__(**kw)
        self.model = model
        self.fallbacks = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"llm:{self.model.name}"

    def rewrite(self, text: str, event: Event, rng: random.Random) -> str:
        keep = required_strings(event)
        if event.kind in (EventKind.FORGET_ALL, EventKind.FORGET_TOPIC, EventKind.FORGET_ITEM):
            return text  # keep forget requests unambiguous
        prompt = self.PROMPT.format(keep=", ".join(f'"{k}"' for k in keep) or "none", text=text)
        out = strip_think(self.model.complete([{"role": "user", "content": prompt}]))
        if out and all(k in out for k in keep):
            return out
        self.fallbacks += 1
        return text
