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
        if c != Category.LEAKAGE:
            assert cats[c] > 0, c


def test_scoring_rules():
    assert mentions("Your sister is called Ilsa.", "ilsa")
    assert not mentions("Ilsabeth", "Ilsa")
    assert mentions("You live in Kraków", "Krakow")
    assert abstains("Sorry, I don't know that.")
    assert abstains("You asked me to forget that.")
    assert not abstains("Your dog is Rex.")
    lo, hi = bootstrap_ci([True] * 50 + [False] * 50)
    assert lo < 0.5 < hi


def test_update_answer_with_stale_value_is_wrong(bench):
    p = next(p for p in bench.probes() if p.category == Category.UPDATE)
    assert score(p, f"It's {p.gold}.").correct
    assert not score(p, f"It's {p.gold}, previously {p.reject[0]}.").correct
    assert score(p, f"It's {p.reject[0]}.").stale


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
