import json
from dataclasses import asdict
from pathlib import Path

import pytest

from engram.bench import dataset
from engram.bench.runner import load_run, rescore, run, save_run
from engram.models.cache import CachedModel, usage
from engram.models.chat import NullModel, ParrotModel
from engram.models.subscription import ClaudeCLI, CLIError, CodexCLI, _transcript
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


class FakeClaude(ClaudeCLI):
    """``ClaudeCLI`` with no CLI behind it: nothing is sent and no subscription is used."""

    def _call(self, system: str, prompt: str) -> str:
        return "I don't know."


class FakeCodex(CodexCLI):
    def _call(self, prompt: str) -> str:
        return "I don't know."


def test_usage_counts_backend_calls_and_names_the_pool(tmp_path):
    msgs = [{"role": "user", "content": "hi"}]
    claude = CachedModel(FakeClaude(), path=tmp_path / "c.sqlite")
    codex = CachedModel(FakeCodex(home=tmp_path), path=None)  # --no-cache: stores nothing
    for model in (claude, codex):
        model.complete(msgs)
        model.complete(msgs)
    assert usage(claude) == {
        "model": "claude:claude-opus-5-5",
        "calls": 1,
        "cache_hits": 1,
        "attempts": 1,
        "pool": "claude-subscription",
    }
    assert usage(codex) == {
        "model": "codex:gpt-6.1-sol",
        "calls": 2,
        "cache_hits": 0,
        "attempts": 2,
        "pool": "chatgpt-subscription",
    }
    assert usage(NullModel()) == {"model": "null"}  # a bare mock reaches no backend


def test_usage_counts_failed_calls_and_retries():
    # One call that succeeds on its third try, then one that fails all three: six invocations.
    replies = iter([CLIError("busy"), CLIError("busy"), "ok", *[CLIError("busy")] * 3])

    class Flaky(ClaudeCLI):
        def _call(self, system: str, prompt: str) -> str:
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

    model = CachedModel(Flaky(retry_delay=0), path=None)
    assert model.complete([{"role": "user", "content": "a"}]) == "ok"
    with pytest.raises(CLIError):
        model.complete([{"role": "user", "content": "b"}])
    drew = usage(model)
    assert (drew["calls"], drew["cache_hits"], drew["attempts"]) == (2, 0, 6)


def test_run_manifest_records_answer_usage(tmp_path, bench):
    model = CachedModel(FakeClaude(), path=tmp_path / "c.sqlite")
    result = run(
        bench, RetrievalMemory(model), leak_targets_per_user=1, workers=4, session_replies=False
    )
    answer = result.manifest["usage"]["answer"]
    assert answer["model"] == "claude:claude-opus-5-5" and answer["pool"] == "claude-subscription"
    # One call per probe and per leak check, counted across the worker threads.
    assert answer["calls"] + answer["cache_hits"] == len(result.scores) + len(result.leaks)
    assert result.manifest["cache"] == {"hits": answer["cache_hits"], "misses": answer["calls"]}
    save_run(result, tmp_path / "run")
    assert load_run(tmp_path / "run").manifest["usage"] == {"answer": answer}
    assert run(bench, OracleMemory(bench), leakage=False).manifest["usage"] == {}


def test_pilot_runs_load_and_rescore():
    # The tracked pilot predates "usage": its runs still load and rescore, and the template
    # writer still rebuilds its dataset bit for bit.
    pilot = Path(__file__).resolve().parent.parent / "results" / "pilot-2026-10-15"
    manifest = json.loads((pilot / "bench-manifest.json").read_text())
    bench = dataset.generate(dataset.BenchConfig(**manifest["config"]))
    assert dataset.content_hash(bench) == manifest["sha256"]
    runs = [load_run(p) for p in sorted(pilot.iterdir()) if p.is_dir()]
    assert runs and all("cache" in r.manifest and "usage" not in r.manifest for r in runs)
    assert all(len(rescore(r, bench).scores) == len(r.scores) for r in runs)


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


def test_saved_runs_never_contain_email_addresses(tmp_path, bench):
    from engram.bench.runner import load_run, save_run

    class Leaky(NullModel):
        def complete(self, messages, *, max_tokens=512):
            return "Your email is someone.real+tag@example.co.uk, I think."

    result = run(bench, RetrievalMemory(Leaky()), leak_targets_per_user=1)
    save_run(result, tmp_path)
    text = (tmp_path / "results.jsonl").read_text() + (tmp_path / "leakage.jsonl").read_text()
    assert "@" not in text and "[redacted-email]" in text
    assert load_run(tmp_path).manifest["redacted_emails"] > 0
