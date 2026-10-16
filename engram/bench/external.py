"""Convert public long-term-memory benchmarks into engram-bench ``Benchmark`` objects.

The converted benchmarks run through the same protocol as the synthetic one (``runner.run``): each
user's sessions are played in order with a context wipe after each, and every probe is asked in its
own no-write session. That is the point of the conversion: LongMemEval and LoCoMo are usually run
with the whole history in the prompt, and here they are run under a forced wipe instead
(research/eval-design.md, "Existing benchmarks we reuse").

Supported sources
-----------------
``load_longmemeval``
    LongMemEval (Wu et al., ICLR 2025, https://github.com/xiaowu0162/LongMemEval). Files from
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned: ``longmemeval_oracle.json``,
    ``longmemeval_s_cleaned.json``, ``longmemeval_m_cleaned.json`` (the older, uncleaned
    ``xiaowu0162/longmemeval`` repo has the same format). One question instance becomes one user.
``load_locomo``
    LoCoMo (Maharana et al., ACL 2024, https://github.com/snap-research/locomo,
    ``data/locomo10.json``). One conversation becomes one user, with all of its QA pairs.

Conversion choices
------------------
* **Session order and dates.** Sessions are sorted chronologically (stable). The LongMemEval
  oracle file is unsorted by design, and the ``_s`` file shuffles sessions within a day. The first
  turn of each session carries a ``[Conversation date: ...]`` line, and LongMemEval questions carry
  ``[Current date: ...]``, because temporal questions cannot be answered without them. Both
  benchmarks give these dates to the reader model in their own protocols. ``Session.day`` counts
  days since the user's first session.
* **Assistant turns (LongMemEval).** The haystack's assistant messages are kept, each as its own
  ``Turn`` whose text starts with ``ASSISTANT_PREFIX``. The system under test still writes its own
  reply to every turn, but it can see, and store, what "it" said at the time. The 56
  ``single-session-assistant`` questions are about exactly those messages. Pass
  ``assistant_turns=False`` to drop them, which is LongMemEval's ``USERONLY`` setting.
* **Speakers (LoCoMo).** LoCoMo is a chat between two people, and its questions are in the third
  person about either of them ("What did Caroline research?"). So both speakers' turns are user
  text, written as ``"Name: message"``. Shared photos become ``[shares a photo: <caption>]``,
  using the BLIP caption the dataset provides.
* **Probes.** Each question is asked once, after the user's last session: ``POST_SLEEP`` if a
  sleep falls after that session, otherwise ``PRE_SLEEP``. Sleeps follow the usual schedule,
  ``sleep_sessions(max sessions, sleep_every)``. The external sources carry no item timelines, so
  ``User.items`` is empty, ``Probe.item_id`` is None, and ``User.canary`` is ``""``. The leakage
  harness has nothing to look for, so run these benchmarks with ``leakage=False``.
* **Categories.** LongMemEval: ``knowledge-update`` maps to UPDATE, ``temporal-reasoning`` to
  TEMPORAL, ``single-session-preference`` to PREFERENCE, any ``question_id`` ending in ``_abs`` to
  ABSTENTION with ``gold=None``, and everything else to FACT_RECALL. LoCoMo, using the numbering
  in ``locomo10.json`` (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial):
  2 maps to TEMPORAL, 5 to ABSTENTION with ``gold=None``, and the rest to FACT_RECALL. An
  adversarial item's ``adversarial_answer`` goes into ``reject``, so stating it counts as a stale
  answer.

Free-form answers and the judge
-------------------------------
engram scores by whole-word string match against ``Probe.accept``. That under-counts on these
benchmarks: LongMemEval answers are often sentences or rubrics ("The user would prefer
responses that ..."), its official metric is a GPT-4o judge, and LoCoMo dates and open-domain
answers are free-form too. The loaders return an ``ExternalBenchmark``, a ``Benchmark`` subclass
whose ``needs_judge`` field holds the ids of probes whose string-match score should be treated as a
lower bound and re-scored by the LLM judge. A probe is flagged if its answer is longer than
``MAX_SPAN_WORDS`` words, or its category is TEMPORAL or PREFERENCE (formats vary), or it is a
LoCoMo open-domain question. Abstention probes are never flagged; they are scored by
``scoring.abstains`` like every other engram abstention probe.

The flag is kept out of ``schema.Probe`` on purpose. ``dataset.save``/``dataset.load`` round-trip
the users unchanged but drop the flag; ``save``/``load`` in this module also write and read it, in
an ``external.json`` sidecar that is checked against the dataset hash.

Answers get some cheap extra ``accept`` variants: numbers also match their word form (``3`` also
accepts ``three``), and ``"X (or Y)"`` also accepts ``X`` and ``Y``. ``gold`` always keeps the
original answer text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from engram.bench import dataset
from engram.bench.dataset import BenchConfig, Benchmark
from engram.bench.probes import sleep_sessions
from engram.bench.schema import Category, Phase, Probe, Session, Turn, User
from engram.bench.scoring import normalize

ASSISTANT_PREFIX = "[Assistant, earlier in this conversation] "
MAX_SPAN_WORDS = 4
SIDECAR = "external.json"

LONGMEMEVAL_CATEGORY = {
    "knowledge-update": Category.UPDATE,
    "temporal-reasoning": Category.TEMPORAL,
    "single-session-preference": Category.PREFERENCE,
    "single-session-user": Category.FACT_RECALL,
    "single-session-assistant": Category.FACT_RECALL,
    "multi-session": Category.FACT_RECALL,
}

LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
LOCOMO_CATEGORY = {
    1: Category.FACT_RECALL,
    2: Category.TEMPORAL,
    3: Category.FACT_RECALL,
    4: Category.FACT_RECALL,
    5: Category.ABSTENTION,
}
_LOCOMO_ALWAYS_JUDGE = {3}  # open-domain answers are inferences, never verbatim spans

_ALWAYS_JUDGE = {Category.TEMPORAL, Category.PREFERENCE}

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
]  # fmt: skip


@dataclass
class ExternalBenchmark(Benchmark):
    """A converted benchmark: a ``Benchmark`` plus the ids of probes that need the LLM judge."""

    needs_judge: set[str] = field(default_factory=set)


# --- shared helpers ------------------------------------------------------------------------------


# json.dumps(ensure_ascii=False) leaves these unescaped, and str.splitlines() (used by
# dataset.load on users.jsonl) splits on them, so a saved dataset would not load back.
_LINE_SEPARATORS = str.maketrans({"\u2028": "\n", "\u2029": "\n", "\x85": "\n"})


def _text(value: Any) -> str:
    """Source text as a clean string: Unicode line separators become newlines."""
    return str(value).translate(_LINE_SEPARATORS).strip()


def accept_variants(answer: str) -> list[str]:
    """The answer itself, plus its number-word form and any ``(or ...)`` alternatives."""
    out = [answer]
    alternatives = re.findall(r"\(or ([^)]+)\)", answer)
    if alternatives:
        out.append(re.sub(r"\s*\(or [^)]+\)", "", answer).strip())
        out.extend(a.strip() for a in alternatives)
    for value in list(out):
        if value.isdigit() and int(value) < len(_NUMBER_WORDS):
            out.append(_NUMBER_WORDS[int(value)])
    return list(dict.fromkeys(v for v in out if v))


def _is_free_form(answer: str) -> bool:
    return len(normalize(answer).split()) > MAX_SPAN_WORDS


def _needs_judge(category: Category, gold: str | None) -> bool:
    if gold is None:
        return False
    return category in _ALWAYS_JUDGE or _is_free_form(gold)


def _final_point(n_sessions: int, sleeps: set[int]) -> tuple[int, Phase, str]:
    last = n_sessions - 1
    if last in sleeps:
        return last, Phase.POST_SLEEP, "b"
    return last, Phase.PRE_SLEEP, "a"


def _dated(turns: list[Turn], when: str) -> list[Turn]:
    if turns:
        turns[0] = Turn(f"[Conversation date: {when}]\n{turns[0].text}")
    return turns


def _days(dates: list[date | None]) -> list[int]:
    """Days since the first parseable date; unparseable dates repeat the previous day."""
    known = [d for d in dates if d is not None]
    if not known:
        return list(range(len(dates)))
    start = min(known)
    out, prev = [], 0
    for d in dates:
        prev = (d - start).days if d is not None else prev
        out.append(prev)
    return out


def _parse(text: str, fmt: str) -> datetime | None:
    try:
        return datetime.strptime(text, fmt)
    except (TypeError, ValueError):
        return None


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finish(
    users: list[User], needs_judge: set[str], sleep_every: int, writer: str
) -> ExternalBenchmark:
    config = BenchConfig(
        users=len(users),
        sessions=max((len(u.sessions) for u in users), default=0),
        seed=0,
        sleep_every=sleep_every,
        late_sleeps=0,
        items_per_user=0,
        abstain_per_user=0,
    )
    return ExternalBenchmark(config, users, writer, needs_judge=needs_judge)


def _schedule(raw_users: list[dict[str, Any]], sleep_every: int) -> set[int]:
    longest = max((len(u["sessions"]) for u in raw_users), default=0)
    return set(sleep_sessions(longest, sleep_every))


# --- LongMemEval ---------------------------------------------------------------------------------


def _lme_date(stamp: str) -> datetime | None:
    # "2023/05/20 (Sat) 02:21": drop the weekday, which strptime cannot check against the date.
    return _parse(re.sub(r"\s*\([A-Za-z]+\)\s*", " ", stamp).strip(), "%Y/%m/%d %H:%M")


def longmemeval_category(instance: dict[str, Any]) -> Category:
    if str(instance["question_id"]).endswith("_abs"):
        return Category.ABSTENTION
    return LONGMEMEVAL_CATEGORY.get(instance["question_type"], Category.FACT_RECALL)


def _lme_sessions(
    instance: dict[str, Any], assistant_turns: bool
) -> list[tuple[datetime | None, str, list[Turn]]]:
    raw = list(zip(instance["haystack_dates"], instance["haystack_sessions"], strict=True))
    parsed = [(_lme_date(stamp), n, stamp, turns) for n, (stamp, turns) in enumerate(raw)]
    # Chronological, stable on ties; unparseable dates keep their original position at the end.
    parsed.sort(key=lambda x: (x[0] is None, x[0] or datetime.min, x[1]))
    out = []
    for when, _, stamp, messages in parsed:
        turns = []
        for m in messages:
            text = _text(m.get("content", ""))
            if not text:
                continue
            if m.get("role") == "assistant":
                if not assistant_turns:
                    continue
                text = ASSISTANT_PREFIX + text
            turns.append(Turn(text))
        out.append((when, stamp, _dated(turns, stamp)))
    return out


def load_longmemeval(
    path: Path | str,
    limit: int | None = None,
    *,
    sleep_every: int = 3,
    assistant_turns: bool = True,
) -> ExternalBenchmark:
    """Convert a LongMemEval JSON file (oracle, ``_s`` or ``_m``) into an engram benchmark.

    Each question instance becomes one user named after its ``question_id``. Its haystack sessions
    become that user's sessions, and the question is a single probe asked after the last session.
    See the module docstring for the conversion choices and for ``needs_judge``.
    """
    path = Path(path)
    instances = _read(path)
    if not isinstance(instances, list):
        raise ValueError(f"{path}: expected a JSON list of LongMemEval instances")
    if limit is not None:
        instances = instances[:limit]

    raw_users = []
    for inst in instances:
        uid = f"lme-{inst['question_id']}"
        raw_users.append(
            {"id": uid, "inst": inst, "sessions": _lme_sessions(inst, assistant_turns)}
        )
    sleeps = _schedule(raw_users, sleep_every)

    users, needs_judge = [], set()
    for raw in raw_users:
        uid, inst = raw["id"], raw["inst"]
        sessions = raw["sessions"]
        if not sessions:
            raise ValueError(f"{path}: {inst['question_id']} has no haystack sessions")
        days = _days([w.date() if w else None for w, _, _ in sessions])
        user = User(id=uid, name=f"LongMemEval {inst['question_id']}", canary="")
        user.sessions = [
            Session(uid, i, days[i], False, turns) for i, (_, _, turns) in enumerate(sessions)
        ]
        category = longmemeval_category(inst)
        gold = None if category == Category.ABSTENTION else _text(inst["answer"])
        ask, phase, timing = _final_point(len(sessions), sleeps)
        question = f"[Current date: {inst['question_date']}] {_text(inst['question'])}"
        probe = Probe(
            f"{uid}-q0", uid, category, question, ask, phase, None,
            gold=gold, accept=accept_variants(gold) if gold else [], timing=timing,
        )  # fmt: skip
        user.probes = [probe]
        if _needs_judge(category, gold):
            needs_judge.add(probe.id)
        users.append(user)
    return _finish(users, needs_judge, sleep_every, f"longmemeval:{path.name}")


# --- LoCoMo --------------------------------------------------------------------------------------


def _locomo_sessions(conv: dict[str, Any]) -> list[tuple[int, str, list[dict[str, Any]]]]:
    """``(number, date_time, turns)`` for every non-empty ``session_N``, in session order.

    Some conversations carry ``session_N_date_time`` keys for sessions that do not exist; those
    are ignored.
    """
    numbers = sorted(int(m.group(1)) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)))
    return [
        (n, str(conv.get(f"session_{n}_date_time", "")), conv[f"session_{n}"])
        for n in numbers
        if conv[f"session_{n}"]
    ]


def _locomo_turn(t: dict[str, Any]) -> Turn | None:
    text = _text(t.get("text", ""))
    caption = _text(t.get("blip_caption") or "")
    if caption:
        text = f"{text} [shares a photo: {caption}]".strip()
    if not text:
        return None
    return Turn(f"{t.get('speaker', '?')}: {text}")


def load_locomo(
    path: Path | str, limit: int | None = None, *, sleep_every: int = 3
) -> ExternalBenchmark:
    """Convert LoCoMo's ``locomo10.json`` into an engram benchmark.

    Each conversation becomes one user, with both speakers' turns as user text prefixed by the
    speaker's name. Every QA pair becomes a probe asked after the last session. See the module
    docstring for the conversion choices and for ``needs_judge``.
    """
    path = Path(path)
    samples = _read(path)
    if not isinstance(samples, list):
        raise ValueError(f"{path}: expected a JSON list of LoCoMo conversations")
    if limit is not None:
        samples = samples[:limit]

    raw_users = [{"sample": s, "sessions": _locomo_sessions(s["conversation"])} for s in samples]
    sleeps = _schedule(raw_users, sleep_every)

    users, needs_judge = [], set()
    for raw in raw_users:
        sample, sessions = raw["sample"], raw["sessions"]
        uid = str(sample["sample_id"])
        conv = sample["conversation"]
        if not sessions:
            raise ValueError(f"{path}: {uid} has no sessions")
        user = User(id=uid, name=f"{conv['speaker_a']} and {conv['speaker_b']}", canary="")
        dates = [_parse(when, "%I:%M %p on %d %B, %Y") for _, when, _ in sessions]
        days = _days([d.date() if d else None for d in dates])
        for i, (_, when, turns) in enumerate(sessions):
            converted = [t for t in map(_locomo_turn, turns) if t is not None]
            user.sessions.append(Session(uid, i, days[i], False, _dated(converted, when)))

        ask, phase, timing = _final_point(len(user.sessions), sleeps)
        for n, qa in enumerate(sample["qa"]):
            raw_cat = int(qa["category"])
            category = LOCOMO_CATEGORY.get(raw_cat, Category.FACT_RECALL)
            if category == Category.ABSTENTION:
                gold, accept = None, []
                adversarial = _text(qa.get("adversarial_answer") or "")
                reject = [adversarial] if adversarial else []
            else:
                gold = _text(qa["answer"])
                accept, reject = accept_variants(gold), []
            probe = Probe(
                f"{uid}-q{n}", uid, category, _text(qa["question"]), ask, phase, None,
                gold=gold, accept=accept, reject=reject, timing=timing,
            )  # fmt: skip
            user.probes.append(probe)
            if gold is not None and (
                raw_cat in _LOCOMO_ALWAYS_JUDGE or _needs_judge(category, gold)
            ):
                needs_judge.add(probe.id)
        users.append(user)
    return _finish(users, needs_judge, sleep_every, f"locomo:{path.name}")


# --- persistence and reporting -------------------------------------------------------------------


def save(bench: ExternalBenchmark, out: Path) -> Path:
    """``dataset.save`` plus an ``external.json`` sidecar holding ``needs_judge``."""
    dataset.save(bench, out)
    sidecar = {
        "writer": bench.writer,
        "sha256": dataset.content_hash(bench),
        "needs_judge": sorted(bench.needs_judge),
    }
    (out / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return out


def load(path: Path) -> ExternalBenchmark:
    """Load a benchmark saved by ``save`` (or by ``dataset.save``, with no judge flags)."""
    bench = dataset.load(path)
    needs_judge: set[str] = set()
    sidecar_path = path / SIDECAR
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("sha256") != dataset.content_hash(bench):
            raise ValueError(f"{path}: {SIDECAR} does not match the dataset hash")
        needs_judge = set(sidecar.get("needs_judge", []))
    return ExternalBenchmark(bench.config, bench.users, bench.writer, needs_judge=needs_judge)


def summarize(bench: Benchmark) -> dict[str, int]:
    """Probe counts per category (in ``Category`` order, categories with no probes omitted)."""
    counts = Counter(p.category for p in bench.probes())
    return {c.value: counts[c] for c in Category if counts[c]}


def judge_ids(bench: Benchmark) -> set[str]:
    """Probe ids that need the LLM judge: empty for a plain (synthetic) ``Benchmark``."""
    return set(getattr(bench, "needs_judge", set()))


__all__ = [
    "ASSISTANT_PREFIX",
    "ExternalBenchmark",
    "accept_variants",
    "judge_ids",
    "load",
    "load_locomo",
    "load_longmemeval",
    "longmemeval_category",
    "save",
    "summarize",
]
