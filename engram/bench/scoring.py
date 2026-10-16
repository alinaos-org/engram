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


def score(probe: Probe, answer: str) -> Score:
    abst = abstains(answer)
    rejects = [r for r in probe.reject if r]
    if probe.gold is None:
        # Never-given, forgotten or incognito: the system must decline and must not reveal the
        # value at all, even framed as history. ``stale`` is reported as the leak rate.
        stale = any(mentions(answer, r) for r in rejects)
        correct = abst and not stale
        ambiguous = False
    else:
        hit = any(mentions(answer, a) for a in probe.accept)
        stale = any(asserts(answer, r) for r in rejects)
        ambiguous = hit and any(mentions(answer, r) for r in rejects)
        # FAMA-style: an answer that also asserts a stale value is wrong, not half-right.
        correct = hit and not stale
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
