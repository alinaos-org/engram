"""Membership inference against per-user adapters (eval-design.md §Forgetting check 2; threat X2).

The attacker holds a user's decrypted adapter and the public base model. Following LoRA-Leak
(Ran et al., arXiv:2507.18302), the pre-trained base is the reference model: a text the adapter
trained on should have lower loss under base+adapter than under the base alone, so the score is

    score(x) = base_loss(x) - adapter_loss(x)        (higher => more likely a member)

Three groups of planted messages are scored against matched non-members:

* **member**: an assert, update or canary message the adapter trained on, about an item the user
  never asked to forget. AUC well above 0.5 means the adapter exposes membership.
* **forgotten**: a trained-on message about an item the user later asked to forget, where a sleep
  has run since the request. Honoured forgetting means AUC ≈ 0.5.
* **incognito**: a message from an incognito session, which must never be trained on. AUC ≈ 0.5.

Each candidate gets a **non-member** twin: the same message with every planted value swapped for a
fresh value from the same slot generator (``engram.bench.world``) that the user never had, so both
sides share a template and a value distribution and differ only in whether the adapter saw them.
Twins whose value cannot be replaced (e.g. a two-option slot the user already used both values of)
are dropped together with their candidate.

Each group is compared with **its own twins**, never with a pooled non-member set. The adapter also
learns templates, so a fresh-value twin of a trained message scores higher than a message whose
template was never trained (incognito slots are disjoint from asserted ones); against a pooled set
that template effect, not membership, decides the AUC.

AUC is the Mann-Whitney statistic; confidence intervals are a stratified percentile bootstrap over
texts. Texts from one user are correlated, so with many users a user-level bootstrap is the more
conservative choice; the pooled CI here is optimistic.

Run this after the benchmark run and **before** anything deletes users (the leakage pass deletes
every user it probes), e.g. ``run(bench, system, leakage=False)`` then ``evaluate(system, bench)``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from engram.bench.dataset import Benchmark
from engram.bench.schema import EventKind, User
from engram.bench.world import SLOT_BY_NAME, make_canary

GROUPS = ("member", "forgotten", "incognito")
_PLANTED = (EventKind.ASSERT, EventKind.UPDATE, EventKind.CANARY)
_FORGETS = (EventKind.FORGET_ITEM, EventKind.FORGET_TOPIC, EventKind.FORGET_ALL)


# -- statistics ------------------------------------------------------------------------------


def auc(pos: list[float], neg: list[float]) -> float:
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie), via midranks (Mann-Whitney U)."""
    if not pos or not neg:
        raise ValueError("auc needs at least one positive and one negative")
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    rank_sum, i = 0.0, 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        midrank = (i + j) / 2 + 1  # 1-based ranks i+1..j+1
        rank_sum += midrank * sum(label for _, label in scored[i : j + 1])
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(
    pos: list[float], neg: list[float], *, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for ``auc`` resampling positives and negatives separately (stratified)."""
    rng = random.Random(seed)
    stats = sorted(
        auc(rng.choices(pos, k=len(pos)), rng.choices(neg, k=len(neg))) for _ in range(n)
    )
    lo = stats[int((alpha / 2) * (n - 1))]
    hi = stats[int((1 - alpha / 2) * (n - 1))]
    return lo, hi


# -- texts -----------------------------------------------------------------------------------


@dataclass
class Sample:
    user_id: str
    group: str  # member | forgotten | incognito | nonmember
    text: str
    item_id: str | None  # None for the canary
    twin: int  # index of the paired sample (candidate <-> non-member)
    base_loss: float | None = None
    adapter_loss: float | None = None

    @property
    def score(self) -> float:
        assert self.base_loss is not None and self.adapter_loss is not None
        return self.base_loss - self.adapter_loss


def _mentions(value: str, texts: list[str]) -> bool:
    pat = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")
    return any(pat.search(t) for t in texts)


def _fresh(sample, taken: set[str], seen: list[str], tries: int = 200) -> str | None:
    for _ in range(tries):
        v = sample()
        if v not in taken and not _mentions(v, seen):
            return v
    return None


def _swap(text: str, mapping: dict[str, str]) -> str | None:
    """Replace every planted value in ``text`` at once; None if one is missing."""
    if not all(_mentions(v, [text]) for v in mapping):
        return None
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile("|".join(r"(?<!\w)" + re.escape(k) + r"(?!\w)" for k in keys))
    return pat.sub(lambda m: mapping[m.group(0)], text)


def build_samples(user: User, logged: list[str], trained: int, *, seed: int = 0) -> list[Sample]:
    """Member / forgotten / incognito candidates for ``user`` and their non-member twins.

    ``logged`` is the user's write-enabled message log in order (``LoRAMemory.logs[user]``) and
    ``trained`` how many of its entries the current adapter was trained on (``trained_upto``).
    """
    rng = random.Random(f"{seed}:{user.id}")
    all_texts = [t.text for s in user.sessions for t in s.turns]

    # Position of each write-enabled turn in the log; it must match the benchmark script.
    log_pos: dict[tuple[int, int], int] = {}
    k = 0
    for s in user.sessions:
        if s.incognito:
            continue
        for j, turn in enumerate(s.turns):
            if k < len(logged):
                if logged[k] != turn.text:
                    raise ValueError(f"{user.id}: log entry {k} does not match the benchmark")
                log_pos[(s.index, j)] = k
            k += 1

    # An item counts as forgotten once a sleep has trained on the log past its forget request.
    forget_pos: dict[tuple[int, str], int] = {}
    for s in user.sessions:
        for j, turn in enumerate(s.turns):
            if turn.event and turn.event.kind in _FORGETS and (s.index, j) in log_pos:
                forget_pos[(s.index, turn.event.kind.value)] = log_pos[(s.index, j)]

    def forgotten(item_id: str) -> bool | None:
        """True if forgotten and consolidated, None if the forget is still pending."""
        for c in user.item(item_id).timeline:
            if c.value is None:
                pos = forget_pos.get((c.session, c.event.value))
                return True if pos is not None and pos < trained else None
        return False

    samples: list[Sample] = []
    for s in user.sessions:
        for j, turn in enumerate(s.turns):
            ev = turn.event
            if ev is None or ev.kind not in _PLANTED or ev.value is None:
                continue
            if s.incognito:
                group = "incognito"
            else:
                pos = log_pos.get((s.index, j))
                if pos is None or pos >= trained:
                    continue  # logged after the last sleep: not in the adapter yet
                if ev.kind == EventKind.CANARY:
                    group = "member"
                else:
                    f = forgotten(ev.item_id or "")
                    if f is None:
                        continue
                    group = "forgotten" if f else "member"

            if ev.kind == EventKind.CANARY:
                gen = lambda: make_canary(rng)  # noqa: E731
                taken = {user.canary}
            else:
                item = user.item(ev.item_id or "")
                slot = SLOT_BY_NAME[item.slot]
                gen = lambda slot=slot: slot.values(rng)  # noqa: E731
                taken = {
                    v for i in user.items if i.slot == item.slot for v in i.values_before(10**9)
                }
            originals = [ev.value] + ([ev.previous] if ev.previous else [])
            mapping: dict[str, str] = {}
            for v in originals:
                new = _fresh(gen, taken | set(mapping.values()), all_texts)
                if new is None:
                    break
                mapping[v] = new
            twin = _swap(turn.text, mapping) if len(mapping) == len(originals) else None
            if twin is None:
                continue
            n = len(samples)
            samples.append(Sample(user.id, group, turn.text, ev.item_id, twin=n + 1))
            samples.append(Sample(user.id, "nonmember", twin, ev.item_id, twin=n))
    return samples


# -- evaluation ------------------------------------------------------------------------------


@dataclass
class GroupAUC:
    auc: float | None
    ci: tuple[float, float] | None
    n_pos: int
    n_neg: int


@dataclass
class MembershipResult:
    model: str
    pooled: dict[str, GroupAUC]
    per_user: dict[str, dict[str, GroupAUC]]
    samples: list[Sample] = field(default_factory=list)


def _group_aucs(
    samples: list[Sample], user: str | None, n_boot: int, seed: int
) -> dict[str, GroupAUC]:
    out = {}
    for g in GROUPS:
        cands = [s for s in samples if s.group == g and user in (None, s.user_id)]
        pos = [s.score for s in cands]
        neg = [samples[s.twin].score for s in cands]
        if pos:
            ci = bootstrap_ci(pos, neg, n=n_boot, seed=seed) if n_boot else None
            out[g] = GroupAUC(auc(pos, neg), ci, len(pos), len(neg))
        else:
            out[g] = GroupAUC(None, None, 0, 0)
    return out


def summarise(
    samples: list[Sample], model: str = "", *, n_boot: int = 1000, seed: int = 0
) -> MembershipResult:
    """AUCs from scored samples (``twin`` indices refer to positions in ``samples``)."""
    per_user = {
        uid: _group_aucs(samples, uid, n_boot, seed)
        for uid in dict.fromkeys(s.user_id for s in samples)
    }
    return MembershipResult(model, _group_aucs(samples, None, n_boot, seed), per_user, samples)


def evaluate(
    system, bench: Benchmark, *, seed: int = 0, n_boot: int = 1000, batch_size: int | None = None
) -> MembershipResult:
    """Score every user that has an adapter in ``system`` (a ``LoRAMemory`` after a run)."""
    samples: list[Sample] = []
    for user in bench.users:
        if system._adapter(user.id) not in system.adapters:
            continue
        mine = build_samples(
            user, list(system.logs.get(user.id, [])), system.trained_upto.get(user.id, 0), seed=seed
        )
        if not mine:
            continue
        texts = [s.text for s in mine]
        adapter = system.sequence_losses(user.id, texts, batch_size=batch_size)
        base = system.base_losses(texts, batch_size=batch_size)
        offset = len(samples)
        for s, a, b in zip(mine, adapter, base, strict=True):
            s.adapter_loss, s.base_loss, s.twin = a, b, s.twin + offset
        samples.extend(mine)
    return summarise(samples, system.name, n_boot=n_boot, seed=seed)


def _fmt(g: GroupAUC) -> str:
    if g.auc is None:
        return f"{'—':>6} {'':>15} {g.n_pos:>5} {g.n_neg:>5}"
    ci = f"[{g.ci[0]:.3f}, {g.ci[1]:.3f}]" if g.ci else ""
    return f"{g.auc:>6.3f} {ci:>15} {g.n_pos:>5} {g.n_neg:>5}"


def format_report(result: MembershipResult, *, per_user: bool = True) -> str:
    head = f"{'':<12} {'group':<10} {'AUC':>6} {'95% CI':>15} {'n_pos':>5} {'n_neg':>5}"
    lines = [
        f"Membership inference (score = base loss - adapter loss), {result.model}",
        "Each group vs its own fresh-value twins. Member AUC = exposure; others should be ≈ 0.5",
        head,
    ]
    for g in GROUPS:
        lines.append(f"{'pooled':<12} {g:<10} {_fmt(result.pooled[g])}")
    if per_user:
        for uid, groups in result.per_user.items():
            for g in GROUPS:
                lines.append(f"{uid:<12} {g:<10} {_fmt(groups[g])}")
    return "\n".join(lines)
