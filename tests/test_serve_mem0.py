import json
import urllib.request

import pytest

from engram.models.chat import ParrotModel
from engram.models.serve import BackgroundServer, _extract_json, complete_openai


def _post(url, body):
    req = urllib.request.Request(
        url + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )  # fmt: skip
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def test_shim_speaks_openai_chat_completions():
    with BackgroundServer(ParrotModel()) as srv:
        out = _post(srv.base_url, {"model": "x", "messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]})  # fmt: skip
        assert out["object"] == "chat.completion"
        assert out["choices"][0]["message"]["content"] == "sys"
        with urllib.request.urlopen(srv.base_url + "/models", timeout=10) as r:
            assert json.load(r)["data"][0]["id"] == "parrot"


def test_shim_json_mode_extracts_object():
    assert _extract_json('Sure! ```json\n{"facts": ["a"]}\n```') == '{"facts": ["a"]}'
    assert _extract_json('prefix {"a": 1} suffix') == '{"a": 1}'

    class Fenced(ParrotModel):
        def complete(self, messages, *, max_tokens=512):
            assert "valid JSON" in messages[0]["content"]
            return 'Here you go:\n```json\n{"ok": true}\n```'

    out = complete_openai(Fenced(), {"messages": [{"role": "user", "content": "x"}],
                                     "response_format": {"type": "json_object"}})  # fmt: skip
    assert json.loads(out["choices"][0]["message"]["content"]) == {"ok": True}


def test_mem0_baseline_scopes_users_and_deletes():
    pytest.importorskip("mem0")
    pytest.importorskip("sentence_transformers")
    from engram.bench import dataset
    from engram.bench.runner import run
    from engram.systems.mem0_memory import Mem0Memory

    bench = dataset.generate(dataset.BenchConfig(users=2, sessions=6, seed=2))
    # infer=False stores raw messages without LLM extraction: tests the plumbing offline.
    system = Mem0Memory(ParrotModel(), infer=False)
    try:
        result = run(bench, system, session_replies=False, leak_targets_per_user=1)
    finally:
        system.close()
    assert sum(s.correct for s in result.scores if s.category == "fact_recall") > 0
    assert not any(s.stale for s in result.scores if s.category == "incognito")
    assert not any(c.leaked for c in result.leaks), [c for c in result.leaks if c.leaked][:2]
