"""Tests for the model-side baselines. Skipped unless the ``ml`` extra is installed; they download
two small public models (a tiny random Llama, ~1 MB, and bge-small, ~130 MB) on first run."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from engram.bench import dataset  # noqa: E402
from engram.bench.runner import run  # noqa: E402
from engram.models.chat import ParrotModel  # noqa: E402
from engram.systems.baselines import RetrievalMemory  # noqa: E402
from engram.systems.dense import Embedder, HybridIndex  # noqa: E402
from engram.systems.lora import LoRAConfig, LoRAMemory  # noqa: E402
from engram.systems.retrieval import BM25, Doc  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_hybrid_finds_paraphrase_that_bm25_misses():
    docs = [
        Doc("I do everything in Helix, so keybinding tips should assume that.", "a"),
        Doc("My dog Rex keeps stealing socks.", "a"),
        Doc("We decided to go with DuckDB for the new service.", "a"),
        Doc("I do everything in Emacs.", "b"),
    ]
    bm25, hybrid = BM25(), HybridIndex(Embedder.shared())
    for d in docs:
        bm25.add(d)
        hybrid.add(d)
    q = "Which editor do I use?"
    assert docs[0] not in bm25.search(q, 1, owner="a")
    top = hybrid.search(q, 1, owner="a")
    assert top == [docs[0]]
    assert all(d.owner == "a" for d in hybrid.search(q, 5, owner="a"))
    assert hybrid.remove(lambda d: d.owner == "a") == 3
    assert hybrid.search(q, 5, owner="a") == []


def test_hybrid_retrieval_baseline_runs_in_parallel():
    bench = dataset.generate(dataset.BenchConfig(users=3, sessions=9, seed=4))
    result = run(
        bench, RetrievalMemory(ParrotModel(), retriever="hybrid"), workers=3, session_replies=False
    )
    assert sum(s.correct for s in result.scores if s.category == "fact_recall") > 0
    assert not any(c.leaked for c in result.leaks)


def test_lora_baseline_trains_per_user_adapters_and_deletes():
    bench = dataset.generate(dataset.BenchConfig(users=2, sessions=6, seed=1))
    system = LoRAMemory(TINY, LoRAConfig(epochs=1, max_new_tokens=8, rank=4))
    result = run(bench, system, session_replies=False, leak_targets_per_user=1)
    assert len(result.scores) == sum(len(u.probes) for u in bench.users)
    assert system.train_steps > 0
    assert system.adapters == {"u_u0000", "u_u0001"}
    assert result.leaks
    system.delete_user("u0000")
    assert system.adapters == {"u_u0001"} and "u0000" not in system.logs
