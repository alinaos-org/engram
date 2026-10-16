"""Membership inference against per-user LoRA adapters (engram.bench.membership)."""

import re

import pytest

from engram.bench import dataset
from engram.bench.membership import Sample, auc, bootstrap_ci, build_samples, summarise
from engram.bench.schema import EventKind


def test_auc_perfect_chance_inverted_and_ties():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auc([1, 2, 3, 4], [1, 2, 3, 4]) == 0.5
    assert auc([1, 1], [1, 1, 1]) == 0.5
    # brute-force pairwise definition on a mixed case with ties
    pos, neg = [0.3, 0.7, 0.7, 0.9, 0.1], [0.2, 0.7, 0.5, 0.1]
    pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    assert auc(pos, neg) == pytest.approx(sum(pairs) / len(pairs))
    with pytest.raises(ValueError):
        auc([], [1.0])


def test_bootstrap_ci_brackets_the_estimate():
    pos, neg = [0.5, 0.9, 0.4, 0.8, 0.7, 0.6], [0.1, 0.45, 0.3, 0.2, 0.65, 0.0]
    lo, hi = bootstrap_ci(pos, neg, n=300)
    assert 0.0 <= lo <= auc(pos, neg) <= hi <= 1.0
    assert bootstrap_ci([1, 2], [0, 0.5], n=50) == (1.0, 1.0)


def test_each_group_is_compared_with_its_own_twins():
    def pair(uid, group, cand, twin):
        return [
            Sample(uid, group, "x", None, twin=0, base_loss=cand, adapter_loss=0.0),
            Sample(uid, "nonmember", "y", None, twin=0, base_loss=twin, adapter_loss=0.0),
        ]

    # members beat their twins; incognito texts score low but so do their own twins.
    rows = [
        *pair("a", "member", 5.0, 4.0),
        *pair("a", "member", 6.0, 4.5),
        *pair("a", "incognito", 1.0, 2.0),
        *pair("b", "incognito", 2.0, 1.0),
        *pair("b", "member", 7.0, 3.0),
    ]
    for i in range(0, len(rows), 2):
        rows[i].twin, rows[i + 1].twin = i + 1, i
    result = summarise(rows, n_boot=0)
    assert result.pooled["member"].auc == 1.0 and result.pooled["member"].n_neg == 3
    assert result.pooled["incognito"].auc == 0.5  # a pooled non-member set would give 0.0
    assert result.per_user["a"]["incognito"].auc == 0.0
    assert result.per_user["b"]["incognito"].auc == 1.0
    assert result.pooled["forgotten"].auc is None


def _logged(user) -> list[str]:
    return [t.text for s in user.sessions if not s.incognito for t in s.turns]


def _skeleton(text: str, values: list[str]) -> str:
    for v in sorted(values, key=len, reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(v) + r"(?!\w)", "<V>", text)
    return text


def test_non_members_are_form_matched_twins():
    bench = dataset.generate(dataset.BenchConfig(users=3, sessions=12, seed=4))
    groups: set[str] = set()
    for user in bench.users:
        logged = _logged(user)
        samples = build_samples(user, logged, trained=len(logged))
        assert samples and len(samples) % 2 == 0
        events = {t.text: t.event for s in user.sessions for t in s.turns if t.event}
        own = {v for i in user.items for c in i.timeline if (v := c.value)} | {user.canary}
        for k, s in enumerate(samples):
            groups.add(s.group)
            twin = samples[s.twin]
            assert twin.twin == k and twin.item_id == s.item_id
            if s.group == "nonmember":
                continue
            ev = events[s.text]
            originals = [ev.value] + ([ev.previous] if ev.previous else [])
            # same template, fresh values, none of which the user ever had
            assert twin.text != s.text
            fresh = _values_in(twin.text, s.text, originals)
            assert not set(fresh) & own
            if s.group == "incognito":
                assert not any(s.text == t for t in logged)
            if ev.kind == EventKind.CANARY:
                assert s.group == "member"
    assert {"member", "nonmember", "incognito"} <= groups


def _values_in(twin: str, text: str, originals: list[str]) -> list[str]:
    pattern = re.escape(_skeleton(text, originals)).replace(re.escape("<V>"), "(.+?)")
    m = re.fullmatch(pattern, twin)
    assert m is not None
    return list(m.groups())


def test_forgotten_needs_a_sleep_after_the_request_and_untrained_is_skipped():
    bench = dataset.generate(dataset.BenchConfig(users=6, sessions=12, seed=2))
    user = next(
        u
        for u in bench.users
        if any(
            t.event and t.event.kind == EventKind.FORGET_ITEM for s in u.sessions for t in s.turns
        )
    )
    logged = _logged(user)
    full = build_samples(user, logged, trained=len(logged))
    assert any(s.group == "forgotten" for s in full)
    # Trained up to (not including) the forget request: the item is pending, so it is neither
    # a member nor forgotten yet.
    forget = next(
        t
        for s in user.sessions
        for t in s.turns
        if t.event and t.event.kind == EventKind.FORGET_ITEM
    )
    item_id = forget.event.item_id
    pending = build_samples(user, logged, trained=logged.index(forget.text))
    assert not any(s.item_id == item_id for s in pending)
    assert any(s.item_id == item_id and s.group == "forgotten" for s in full)
    # Before any training nothing is a member or forgotten; only incognito texts remain.
    none = build_samples(user, logged, trained=0)
    assert {s.group for s in none} <= {"incognito", "nonmember"}
    with pytest.raises(ValueError):
        build_samples(user, ["something else", *logged[1:]], trained=len(logged))


def test_end_to_end_on_tiny_lora():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from engram.bench.membership import evaluate, format_report
    from engram.bench.runner import run
    from engram.systems.lora import LoRAConfig, LoRAMemory

    bench = dataset.generate(dataset.BenchConfig(users=2, sessions=6, seed=1))
    system = LoRAMemory(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        LoRAConfig(epochs=1, max_new_tokens=4, rank=4),
    )
    run(bench, system, leakage=False, session_replies=False)
    texts = ["My dog Rex keeps stealing socks.", "I live in Porto."]
    base = system.base_losses(texts)
    adapted = system.sequence_losses("u0000", texts, batch_size=1)
    assert len(base) == len(adapted) == 2 and all(x > 0 for x in base + adapted)
    # batching and padding do not change per-text losses
    assert system.base_losses(texts, batch_size=1) == pytest.approx(base, rel=1e-4)
    with pytest.raises(KeyError):
        system.sequence_losses("nobody", texts)

    result = evaluate(system, bench, n_boot=100)
    for i, smp in enumerate(result.samples):  # twins stay paired after pooling users
        twin = result.samples[smp.twin]
        assert twin.twin == i and twin.user_id == smp.user_id
        assert (smp.group == "nonmember") != (twin.group == "nonmember")
    assert set(result.per_user) == {"u0000", "u0001"}
    member = result.pooled["member"]
    assert member.auc is not None and 0.0 <= member.auc <= 1.0
    assert member.ci is not None and member.ci[0] <= member.auc <= member.ci[1]
    for groups in result.per_user.values():
        for g in groups.values():
            assert g.auc is None or 0.0 <= g.auc <= 1.0
    report = format_report(result)
    assert "pooled" in report and "member" in report


def test_cli_membership_flag(tmp_path, capsys):
    from engram.cli import main as bench_main

    data, out = tmp_path / "b", tmp_path / "r"
    bench_main(["generate", "--out", str(data), "--users", "2", "--sessions", "6", "--seed", "1"])
    bench_main(["run", "--bench", str(data), "--system", "b4", "--base-model",
                "hf-internal-testing/tiny-random-LlamaForCausalLM", "--lora-epochs", "1",
                "--membership", "--no-session-replies", "--leak-targets", "1",
                "--out", str(out)])  # fmt: skip
    assert (out / "membership.json").exists()
    assert "member" in capsys.readouterr().out
