from collections import Counter

import pytest

from engram.bench import dataset
from engram.bench.judge import cohen_kappa
from engram.bench.runner import apply_floor, run
from engram.bench.schema import Category, EventKind, Phase
from engram.bench.scoring import abstains, bootstrap_ci, mentions, score
from engram.models.chat import NullModel, ParrotModel
from engram.systems.baselines import (
    FullContext,
    LeakyRetrieval,
    NoMemory,
    OracleMemory,
    RetrievalMemory,
)
from engram.systems.retrieval import BM25, Doc


@pytest.fixture(scope="module")
def bench():
    return dataset.generate(dataset.BenchConfig(users=6, sessions=15, seed=7))


def test_generation_is_deterministic(bench):
    again = dataset.generate(dataset.BenchConfig(users=6, sessions=15, seed=7))
    assert dataset.content_hash(again) == dataset.content_hash(bench)
    other = dataset.generate(dataset.BenchConfig(users=6, sessions=15, seed=8))
    assert dataset.content_hash(other) != dataset.content_hash(bench)


def test_roundtrip(tmp_path, bench):
    dataset.save(bench, tmp_path)
    loaded = dataset.load(tmp_path)
    assert dataset.content_hash(loaded) == dataset.content_hash(bench)
    (tmp_path / "users.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        dataset.load(tmp_path)


def test_world_invariants(bench):
    canaries = [u.canary for u in bench.users]
    assert len(set(canaries)) == len(canaries)
    for user in bench.users:
        for item in user.items:
            sessions = [c.session for c in item.timeline]
            assert sessions == sorted(sessions)
            assert item.timeline[0].event == EventKind.ASSERT
            # Nothing happens to an item after it is forgotten.
            forgets = [i for i, c in enumerate(item.timeline) if c.value is None]
            assert not forgets or forgets[0] == len(item.timeline) - 1
            if item.incognito_only:
                session = user.sessions[item.first_session()]
                assert session.incognito
        # Every planted event appears in exactly one scripted turn.
        planted = [t.event for s in user.sessions for t in s.turns if t.event]
        assert sum(e.kind == EventKind.CANARY for e in planted) == 1


def test_probe_gold_follows_timeline(bench):
    for user in bench.users:
        for p in user.probes:
            if p.item_id is None or p.category == Category.TEMPORAL:
                continue
            item = user.item(p.item_id)
            expected = None if item.incognito_only else item.value_at(p.ask_after_session)
            assert p.gold == expected, p
            if p.category == Category.UPDATE:
                assert p.reject and p.gold not in p.reject


def test_all_categories_present():
    big = dataset.generate(dataset.BenchConfig(users=20, sessions=30, seed=0))
    cats = Counter(p.category for p in big.probes())
    for c in Category:
        if c not in (Category.LEAKAGE, Category.APPLICATION):  # leakage: harness; app: opt-in
            assert cats[c] > 0, c


def test_scoring_rules():
    assert mentions("Your sister is called Ilsa.", "ilsa")
    assert not mentions("Ilsabeth", "Ilsa")
    assert mentions("You live in Kraków", "Krakow")
    assert abstains("Sorry, I don't know that.")
    assert abstains("You asked me to forget that.")
    assert not abstains("Your dog is Rex.")
    assert abstains("I don\u2019t know which editor you use.")
    assert abstains("You didn't mention where you work.")
    assert abstains("User asked to forget city; don't reveal.")
    assert abstains("You asked me not to keep your city, so I'm not going to share it.")
    lo, hi = bootstrap_ci([True] * 50 + [False] * 50)
    assert lo < 0.5 < hi


def test_update_answer_with_stale_value_is_wrong(bench):
    p = next(p for p in bench.probes() if p.category == Category.UPDATE)
    old = p.reject[0]
    assert score(p, f"It's {p.gold}.").correct
    assert score(p, f"It's {p.gold}. They took over from {old}.").correct
    assert score(p, f"You used to say {old}, but now it's {p.gold}.").correct
    assert score(p, f"It's {p.gold}, previously {old}.").ambiguous
    assert score(p, f"{p.gold}. You switched to it after {old} went downhill.").correct
    assert not score(p, f"It's {old}. Or maybe {p.gold}.").correct
    assert score(p, f"It's {old}.").stale and not score(p, f"It's {old}.").correct


def test_oracle_scores_perfectly(bench):
    result = run(bench, OracleMemory(bench), leakage=False)
    assert len(result.scores) == sum(len(u.probes) for u in bench.users)
    assert all(s.correct for s in result.scores), [s for s in result.scores if not s.correct][:3]


def test_null_model_only_gets_abstention_right(bench):
    result = run(bench, NoMemory(NullModel()), leakage=False)
    for s in result.scores:
        assert s.correct == (s.category in {"abstention", "forget", "incognito"})


def test_retrieval_recalls_and_respects_incognito(bench):
    result = run(bench, RetrievalMemory(ParrotModel()), leakage=False)
    by = Counter((s.category, s.correct) for s in result.scores)
    # The parrot echoes retrieved notes, so recall works when retrieval does...
    assert by[("fact_recall", True)] > 0
    # ...and incognito sessions were never written, so their values never come back.
    assert not any(s.stale for s in result.scores if s.category == "incognito")
    # Naive retrieval does not honour forget requests: the forget category must catch that.
    assert any(s.stale for s in result.scores if s.category == "forget")


def test_full_context_respects_incognito(bench):
    result = run(bench, FullContext(ParrotModel()), leakage=False)
    assert not any(s.stale for s in result.scores if s.category == "incognito")
    assert sum(s.correct for s in result.scores if s.category == "fact_recall") > 0


def test_probes_never_written_to_memory(bench):
    system = RetrievalMemory(ParrotModel())
    run(bench, system, leakage=False)
    probe_questions = {p.question for p in bench.probes()}
    assert not any(d.text in probe_questions for d in system.index.docs)


def test_leakage_harness_passes_gated_and_catches_leaky(bench):
    gated = run(bench, RetrievalMemory(ParrotModel()))
    assert gated.leaks and not any(c.leaked for c in gated.leaks)
    leaky = run(bench, LeakyRetrieval(ParrotModel()))
    assert any(c.leaked for c in leaky.leaks)


def test_deletion_checks(bench):
    gated = run(bench, RetrievalMemory(ParrotModel()))
    deleted = [c for c in gated.leaks if c.kind == "deleted"]
    assert deleted and not any(c.leaked for c in deleted)

    class NoDelete(RetrievalMemory):
        def delete_user(self, user_id):  # a system that "forgets" to delete
            pass

    broken = run(bench, NoDelete(ParrotModel()))
    assert any(c.leaked for c in broken.leaks if c.kind == "deleted")


def test_option_lists_are_neither_recall_nor_leak(bench):
    gold_probe = next(p for p in bench.probes() if p.gold)
    other = "zzz-alternative"
    hedge = f"I don't know - {gold_probe.gold} or {other}?"
    assert not score(gold_probe, hedge).correct
    forgot = next(p for p in bench.probes() if p.category.value == "incognito")
    value = forgot.reject[0]
    assert score(forgot, f"I don't know. Do you want {value} or {other}?").correct
    assert not score(forgot, f"You told me {value}.").correct


def test_floor_drops_probes_solved_without_memory(bench):
    oracle = run(bench, OracleMemory(bench), leakage=False).scores
    kept = apply_floor(oracle, oracle)
    assert all(s.category in {"abstention", "forget", "incognito"} for s in kept)


def test_bm25_owner_gating():
    idx = BM25()
    idx.add(Doc("my dog is called Rex", "a"))
    idx.add(Doc("my dog is called Fido", "b"))
    assert [d.owner for d in idx.search("dog called", owner="a")] == ["a"]
    assert idx.remove(lambda d: d.owner == "a") == 1
    assert [d.owner for d in idx.search("dog")] == ["b"]


def test_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, False, True, False], [False, True, False, True]) < 0


def test_phase_values_roundtrip(bench):
    assert {p.phase for p in bench.probes()} <= set(Phase)


def test_application_probes_opt_in_and_scoring():
    plain = dataset.generate(dataset.BenchConfig(users=6, sessions=15, seed=7))
    assert not any(p.category == Category.APPLICATION for p in plain.probes())
    big = dataset.generate(
        dataset.BenchConfig(users=20, sessions=30, seed=0, application_probes=True)
    )
    apps = [p for p in big.probes() if p.category == Category.APPLICATION]
    assert apps
    code = next(p for p in apps if p.accept[0].startswith("lang:"))
    lang = code.accept[0][5:]
    assert score(code, f"Here you go:\n```{lang.lower()}\nprint(1)\n```").correct
    assert not score(code, "Here's how to go about it:\n```cobol\nX\n```").correct
    assert score(code, lang).correct  # no code block: plain mention counts (oracle)
    unit = next((p for p in apps if p.question.startswith("Roughly")), None)
    if unit:
        word = unit.accept[0]
        assert score(unit, f"About 200 {word}.").correct
    oracle = run(big, OracleMemory(big), leakage=False)
    assert all(s.correct for s in oracle.scores if s.category == "application")


def test_long_writer_keeps_every_planted_value():
    import json as _json
    import re as _re

    from engram.bench.writer import LongWriter

    class FakeWriter(NullModel):
        """Writes N long messages and drops every second fact, to exercise the fallback."""

        def complete(self, messages, *, max_tokens=512):
            prompt = messages[-1]["content"]
            n = int(_re.search(r"Write (\d+) user messages", prompt).group(1))
            facts = _re.findall(r"^- (.+?)  \(exact strings", prompt, flags=_re.M)
            msgs = [f"Working on it, message {i}. " + "lorem " * 50 for i in range(n)]
            for i, fact in enumerate(facts[::2]):
                msgs[i % n] += " By the way: " + fact
            return "```json\n" + _json.dumps({"messages": msgs}) + "\n```"

    writer = LongWriter(FakeWriter(), workers=3)
    bench = dataset.generate(dataset.BenchConfig(users=3, sessions=9, seed=11), writer)
    for user in bench.users:
        text = " ".join(t.text for s in user.sessions for t in s.turns)
        for item in user.items:
            for change in item.timeline:
                if change.value:
                    assert change.value in text
        planted = [t.event for s in user.sessions for t in s.turns if t.event]
        assert sum(e.kind == EventKind.CANARY for e in planted) == 1
        assert all(len(s.turns) >= 8 for s in user.sessions)
    assert writer.fallbacks > 0 and writer.failed_sessions == 0
    oracle = run(bench, OracleMemory(bench), leakage=False)
    assert all(s.correct for s in oracle.scores)
