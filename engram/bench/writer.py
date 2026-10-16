"""Render a user's planned events into scripted sessions.

``TemplateWriter`` is deterministic and needs no model; CI and unit tests use it. ``LLMWriter``
rewrites each planted message into natural, varied text with a chat model, and falls back to the
template whenever the rewrite drops a value the probes depend on.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

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


WORK_TOPICS = [
    "debugging a flaky integration test in CI",
    "designing the database schema for a billing service",
    "drafting a performance review for a direct report",
    "planning a zero-downtime migration off a legacy queue",
    "reviewing a large pull request that refactors authentication",
    "writing the postmortem for last week's outage",
    "choosing between two observability vendors",
    "preparing slides for a quarterly roadmap review",
    "tuning a slow analytics query",
    "onboarding a new engineer to the codebase",
    "negotiating scope with a product manager",
    "designing rate limiting for a public API",
    "writing a design doc for a caching layer",
    "planning an offsite and its agenda",
    "investigating a memory leak in a long-running worker",
    "setting up feature flags for a gradual rollout",
    "estimating a project for a client proposal",
    "reorganising the team's on-call rotation",
]


class LongWriter(TemplateWriter):
    """Long, natural sessions written by a chat model (gate G1 in research/phase-0-plan.md).

    Each session is 12-20 user messages about a realistic work topic (about 5-7K tokens), so a
    30-session history of user text alone exceeds a 128K context. The session's planted facts
    are woven in, ideally in passing. A 2-user prototype at 8-14 messages averaged ~1.5K tokens
    per session (research/phase-0-plan.md, G1), which was too short. Safeguards keep the gold
    answers valid: every planted value must appear verbatim in some message (otherwise the
    template sentence is inserted as its own message and counted in ``fallbacks``); forget
    requests and canaries are always inserted verbatim; the prompt forbids other personal details.
    """

    PROMPT = """Write the USER's side of one long chat session with an AI work assistant.
The user is a software professional working on: {topic}.
Write {n} user messages, each {words} words, that read like a real working session: questions,
pasted error messages or snippets, follow-ups, reactions to (unseen) assistant answers.

Weave these facts about the user into the session naturally, in passing where possible (for
example while explaining a scheduling constraint or giving context), each in a different message:
{facts}
Each quoted string must appear EXACTLY as written (same spelling and capitalisation).

Do not mention any other personal details about the user (no other names, places, pets,
employers, health, diet, preferences or decisions) beyond the facts above.

Reply with JSON only: {{"messages": ["first message", "second message", ...]}}"""

    def __init__(
        self,
        model: ChatModel,
        *,
        turns: tuple[int, int] = (12, 20),
        words: str = "250 to 450",
        workers: int = 4,
    ) -> None:
        super().__init__()
        self.model = model
        self.turns = turns
        self.words = words
        self.workers = workers
        self.fallbacks = 0
        self.failed_sessions = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"long:{self.model.name}"

    def _messages(self, prompt: str) -> list[str] | None:
        from engram.models.serve import _extract_json

        raw = strip_think(
            self.model.complete(
                [
                    {"role": "system", "content": "You write realistic synthetic test data."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8000,
            )
        )
        try:
            msgs = json.loads(_extract_json(raw)).get("messages")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(msgs, list) or not all(isinstance(m, str) and m.strip() for m in msgs):
            return None
        return [m.strip() for m in msgs]

    def write(self, user: User, plan: Plan, rng: random.Random) -> list[Session]:
        # Draw every random choice up front so the plan stays deterministic while the model
        # calls run in parallel.
        specs = []
        for s, day in enumerate(plan.days):
            planted, verbatim = [], []
            for event in plan.events[s]:
                text = template_text(event, user, rng)
                if event.kind in (
                    EventKind.CANARY, EventKind.FORGET_ITEM, EventKind.FORGET_TOPIC,
                    EventKind.FORGET_ALL,
                ):  # fmt: skip
                    verbatim.append(Turn(text, event))
                else:
                    planted.append((event, text, required_strings(event)))
            n = rng.randint(*self.turns)
            topic = rng.choice(WORK_TOPICS)
            positions = [rng.random() for _ in range(len(planted) + len(verbatim))]
            specs.append((s, day, planted, verbatim, n, topic, positions))

        def build(spec) -> Session:
            s, day, planted, verbatim, n, topic, positions = spec
            facts = (
                "\n".join(
                    f"- {text}  (exact strings: {', '.join(repr(r) for r in req)})"
                    for _, text, req in planted
                )
                or "- (none in this session; keep it purely about work)"
            )
            prompt = self.PROMPT.format(topic=topic, n=n, words=self.words, facts=facts)
            msgs = self._messages(prompt)
            if msgs is None:
                with self._lock:
                    self.failed_sessions += 1
                msgs = [FILLER[(s + i) % len(FILLER)] for i in range(3)]
            turns = [Turn(m) for m in msgs]
            pos = iter(positions)
            for event, text, req in planted:
                where = None
                for i, t in enumerate(turns):
                    if t.event is None and all(r in t.text for r in req):
                        where = i
                        break
                p = next(pos)
                if where is None:
                    with self._lock:
                        self.fallbacks += 1
                    turns.insert(int(p * (len(turns) + 1)), Turn(text, event))
                else:
                    turns[where].event = event
            for turn in verbatim:
                turns.insert(int(next(pos) * (len(turns) + 1)), turn)
            return Session(user.id, s, day, incognito=s in plan.incognito_sessions, turns=turns)

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(build, specs))
