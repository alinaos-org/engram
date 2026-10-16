from dataclasses import asdict

import pytest

from engram.bench import dataset
from engram.bench.runner import run
from engram.models.cache import CachedModel
from engram.models.chat import NullModel, ParrotModel
from engram.models.subscription import _transcript
from engram.systems.baselines import FullContext, LeakyRetrieval, OracleMemory, RetrievalMemory


@pytest.fixture(scope="module")
def bench():
    return dataset.generate(dataset.BenchConfig(users=5, sessions=12, seed=3))


def _key(result):
    return (
        sorted((s.probe_id, s.correct, s.stale) for s in result.scores),
        sorted((c.kind, c.loaded_user, c.target_user, c.question, c.leaked) for c in result.leaks),
    )


@pytest.mark.parametrize("make", [RetrievalMemory, FullContext])
def test_parallel_and_observe_match_sequential(bench, make):
    seq = run(bench, make(ParrotModel()))
    par = run(bench, make(ParrotModel()), workers=4, session_replies=False)
    assert _key(seq) == _key(par)


def test_parallel_leaky_control_still_caught(bench):
    # In parallel mode each user's main run has a private instance, so the leaky control only
    # leaks in the leakage pass, which rebuilds one shared instance. It must still be caught.
    seq = run(bench, LeakyRetrieval(ParrotModel()))
    par = run(bench, LeakyRetrieval(ParrotModel()), workers=4, session_replies=False)
    assert _key(seq)[1] == _key(par)[1]
    assert any(c.leaked for c in par.leaks)


def test_parallel_oracle(bench):
    result = run(bench, OracleMemory(bench), workers=3, session_replies=False)
    assert all(s.correct for s in result.scores)


def test_observe_mode_makes_no_session_calls(bench):
    calls = []

    class Counting(NullModel):
        def complete(self, messages, *, max_tokens=512):
            calls.append(1)
            return super().complete(messages, max_tokens=max_tokens)

    run(bench, RetrievalMemory(Counting()), leakage=False, session_replies=False)
    assert len(calls) == sum(len(u.probes) for u in bench.users)


def test_cache_hits_on_repeat(tmp_path):
    model = CachedModel(ParrotModel(), path=tmp_path / "c.sqlite")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    first = model.complete(msgs)
    assert model.complete(msgs) == first
    assert (model.hits, model.misses) == (1, 1)
    again = CachedModel(ParrotModel(), path=tmp_path / "c.sqlite")
    again.complete(msgs)
    assert again.hits == 1


def test_transcript_flattening():
    system, prompt = _transcript(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert system == "be brief"
    assert "User: a" in prompt and "Assistant: b" in prompt and prompt.endswith("c")
    assert _transcript([{"role": "user", "content": "only"}]) == ("", "only")


def test_manifest_records_parallel_settings(bench):
    result = run(bench, RetrievalMemory(ParrotModel()), workers=2, session_replies=False)
    assert result.manifest["workers"] == 2 and result.manifest["session_replies"] is False
    assert asdict(bench.config) == result.manifest["bench"]["config"]
