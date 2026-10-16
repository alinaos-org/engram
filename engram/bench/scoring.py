"""Score answers to probes, and summarise results with bootstrap confidence intervals.

Scoring is deterministic string matching wherever possible, so results don't depend on a judge.
Matching uses exact values only, never a judge of helpfulness: MemLeak (NeurIPS 2026 workshop)
found contaminated answers are often rated *more* helpful.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from engram.bench.schema import Probe

ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't|dont) (?:know|have|recall|remember)",
    r"\b(?:you )?(?:haven't|have not|never) (?:told|mentioned|said|shared)",
    r"\bno (?:record|information|memory)\b",
    r"\b(?:not sure|unsure)\b",
    r"\b(?:asked me to|you wanted me to) forget\b",
    r"\b(?:asked|requested|wants?|told) (?:me )?"
    r"(?:to forget|not to (?:keep|remember|store|share))\b",
    r"\b(?:i'm|i am|i will|i'll) not (?:going to )?(?:share|reveal|repeat)\b",
    r"\bi (?:can't|cannot|can not) (?:recall|remember|say)\b",
    r"\bi'm not aware\b",
    r"\bnot something (?:i|you)",
    r"\b(?:you )?(?:didn't|did not|never) (?:mention|tell|say|share)",
    r"\bnot enough information\b",
    r"\binformation (?:provided |given )?is not enough\b",
    r"\bnothing (?:saved|stored|on file|recorded)\b",
    r"\bi have no (?:record|information|details|memory)\b",
]
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.I)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def mentions(answer: str, value: str) -> bool:
    """True if ``value`` appears in ``answer`` as whole words (case- and accent-insensitive)."""
    if not value:
        return False
    a, v = f" {normalize(answer)} ", normalize(value)
    return f" {v} " in a


def abstains(answer: str) -> bool:
    text = answer.replace("\u2019", "'").replace("\u2018", "'")  # curly apostrophes
    return bool(_ABSTAIN_RE.search(text))


# Words that frame a mentioned value as history rather than as the current answer.
_PAST_CUE = re.compile(
    r"\b(used to|previously|formerly|before|earlier|originally|no longer|not any more|"
    r"anymore|instead of|rather than|replac\w*|took over from|switched from|moved from|"
    r"changed from|renamed from|left|was|were|old|ex-?\w*|from)\b",
    re.I,
)
_CUE_WINDOW = 40  # characters either side of the stale value


def asserts(answer: str, value: str) -> bool:
    """True if ``value`` is mentioned and at least one mention is *not* framed as the past.

    "Your manager is Lomor; they took over from Lois" mentions Lois only as history, so it does
    not assert Lois. A heuristic: answers that mention both the gold and a stale value are also
    flagged ``ambiguous`` so a judge (or a human) can audit them.
    """
    if not mentions(answer, value):
        return False
    a, v = normalize(answer), normalize(value)
    start = 0
    while (i := a.find(v, start)) != -1:
        window = a[max(0, i - _CUE_WINDOW) : i] + " " + a[i + len(v) : i + len(v) + _CUE_WINDOW]
        if not _PAST_CUE.search(window):
            return True
        start = i + len(v)
    return False


@dataclass
class Score:
    probe_id: str
    user_id: str
    category: str
    timing: str
    correct: bool
    stale: bool  # asserted an outdated or forgotten value (for forget/incognito: revealed it)
    abstained: bool
    answer: str
    ambiguous: bool = False  # mentioned both the gold and a stale value; worth auditing


def first_index(answer: str, value: str) -> int:
    """Position of the first whole-word mention of ``value`` in the normalised answer, or -1."""
    a, v = f" {normalize(answer)} ", normalize(value)
    i = a.find(f" {v} ")
    return i


def only_as_option(answer: str, value: str) -> bool:
    """True if every mention of ``value`` sits in an "A or B" option list.

    "I don't know - metric or imperial?" names both units without claiming either, so it is not a
    recall (for a gold value) and not a leak (for a forgotten or incognito value).
    """
    a, v = normalize(answer), re.escape(normalize(value))
    mentions_ = list(re.finditer(rf"(?<!\w){v}(?!\w)", a))
    if not mentions_:
        return False
    for m in mentions_:
        before, after = a[max(0, m.start() - 12) : m.start()], a[m.end() : m.end() + 12]
        if not (re.search(r"\bor\s*$", before) or re.match(r"^\s*or\b", after)):
            return False
    return True


_FENCE = re.compile(r"```[ \t]*([A-Za-z][\w+#-]*)")


def fence_languages(answer: str) -> set[str]:
    """Language tags of the answer's fenced code blocks, lower-cased."""
    return {m.group(1).lower() for m in _FENCE.finditer(answer)}


def score(probe: Probe, answer: str) -> Score:
    abst = abstains(answer)
    rejects = [r for r in probe.reject if r]
    langs = fence_languages(answer)

    def said(value: str) -> bool:
        if value.startswith("lang:"):
            # Code-language preference: judged by the code block's tag, so English words that are
            # also language names ("go about it") don't count. Without any fenced code, fall back
            # to a plain mention of the language.
            name = value[5:]
            return name.lower() in langs if langs else said(name)
        # A value only offered as an option next to "I don't know" was not stated.
        return mentions(answer, value) and not (abst and only_as_option(answer, value))

    if probe.gold is None:
        # Never-given, forgotten or incognito: the system must decline and must not reveal the
        # value at all, even framed as history. ``stale`` is reported as the leak rate.
        stale = any(said(r) for r in rejects)
        correct = abst and not stale
        ambiguous = False
    else:
        hits = [a for a in probe.accept if said(a)]
        lang_probe = any(a.startswith("lang:") for a in probe.accept)
        if lang_probe:
            # Wrong language in the code block is stale; history framing doesn't apply to code.
            stale = any(said(r) for r in rejects)
            ambiguous = bool(hits) and stale
        else:
            gold_at = min((first_index(answer, a) for a in hits), default=-1)
            # A stale value counts as asserted if it is not framed as history *and* it comes
            # before the current value (or the current value is missing): "Bistro Feset. You
            # switched to it after Bistro Ullio went downhill" asserts Feset, not Ullio.
            stale = any(
                asserts(answer, r) and (gold_at < 0 or first_index(answer, r) < gold_at)
                for r in rejects
            )
            ambiguous = bool(hits) and any(mentions(answer, r) for r in rejects)
        # FAMA-style: an answer that also asserts a stale value is wrong, not half-right.
        correct = bool(hits) and not stale
    return Score(
        probe.id,
        probe.user_id,
        probe.category.value,
        probe.timing,
        correct,
        stale,
        abst,
        answer,
        ambiguous,
    )


@dataclass
class Summary:
    category: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    stale_rate: float


def bootstrap_ci(
    values: list[bool], *, iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean of booleans."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iters))
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def summarise(scores: list[Score], by: str = "category") -> list[Summary]:
    groups: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        groups[getattr(s, by)].append(s)
        groups["ALL"].append(s)
    out = []
    for key in sorted(groups, key=lambda k: (k == "ALL", k)):
        g = groups[key]
        vals = [s.correct for s in g]
        lo, hi = bootstrap_ci(vals)
        out.append(
            Summary(key, len(g), sum(vals) / len(g), lo, hi, sum(s.stale for s in g) / len(g))
        )
    return out


def format_table(summaries: list[Summary], title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{'category':<14} {'n':>6} {'acc':>7} {'95% CI':>15} {'stale':>7}")
    for s in summaries:
        lines.append(
            f"{s.category:<14} {s.n:>6} {s.accuracy:>7.3f} "
            f"{f'[{s.ci_low:.3f}, {s.ci_high:.3f}]':>15} {s.stale_rate:>7.3f}"
        )
    return "\n".join(lines)
