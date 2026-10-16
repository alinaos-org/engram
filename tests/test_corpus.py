"""Phase 1 corpus rebuild (engram/corpus). No network, no model downloads."""

import json
import re
from pathlib import Path

import pytest

from engram.bench import dataset
from engram.corpus import bios, card, chunk, decontam, episodes, fetch, llm_probes
from engram.corpus import sources as src
from engram.corpus.build import BuildError, Context, parse_stages, run
from engram.corpus.cli import main
from engram.corpus.io import read_jsonl
from engram.corpus.tokenize import WhitespaceTokenizer, get_tokenizer

FIX = Path(__file__).parent / "fixtures"
CORPUS = FIX / "corpus"
TOK = WhitespaceTokenizer()
BENCH = {"users": 3, "sessions": 8, "seed": 0}


@pytest.fixture(scope="module")
def ref():
    return decontam.build_reference(
        bench=BENCH,
        external=[
            CORPUS / "eval_tiny.json",
            FIX / "longmemeval_tiny.json",
            FIX / "locomo_tiny.json",
            CORPUS / "absent.json",
        ],
    )


@pytest.fixture(scope="module")
def bench_users():
    return dataset.generate(dataset.BenchConfig(**BENCH)).users


# -- bios --------------------------------------------------------------------------------------


def test_bios_are_deterministic():
    a = bios.generate(0, 2, people=15)
    assert a == bios.generate(0, 2, people=15)
    assert a[0][0]["text"] != bios.generate(1, 2, people=15)[0][0]["text"]
    # a document depends only on (seed, index), not on how many documents are generated
    assert bios.generate(0, 1, people=15)[0][0] == a[0][0]


def test_bio_probes_are_exact():
    doc, probes = bios.generate_doc(3, 0, people=30, p_update=0.5, p_extra=0.5)
    text = doc["text"]
    kinds = {p["type"] for p in probes}
    assert {"fact", "update", "abstain", "indirect"} <= kinds
    for p in probes:
        for f in p["facts"]:
            assert f["value"] in text[f["start"] : f["end"]]
        if p["type"] == "update":
            values = [f["value"] for f in p["facts"]]
            assert len(set(values)) == len(values) >= 2
            assert [f["start"] for f in p["facts"]] == sorted(f["start"] for f in p["facts"])
        if p["type"] == "indirect":
            assert p["accept"] == [p["facts"][0]["value"]]
            assert p["facts"][0]["value"] in text[p["facts"][0]["start"] : p["facts"][0]["end"]]
        if p["type"] == "abstain":
            assert p["facts"] == [] and p["anchor"]
            start = p["anchor"][0]
            end = text.find("\n\n", start)
            paragraph = text[start : end if end > 0 else len(text)]
            pool = bios._INSTRUMENTS if p["attribute"] == "instrument" else ["mentor", "trained"]
            assert not any(v in paragraph for v in pool), (p["question"], paragraph)
    # the person named in each question is introduced in the document
    for p in probes:
        name = re.search(r"(?:was|did|does|mentored) (\w+ \w+)", p["question"])
        if name and p["type"] != "indirect":
            assert name.group(1) in text


def test_bios_disjoint_from_engram_bench():
    docs, probes = bios.generate(0, 3, people=40)
    assert bios.check_disjoint(docs, probes) == []
    firsts, lasts = bios.bench_name_pools()
    for p in probes:
        for word in re.findall(r"[A-Z][a-z]+", p["question"]):
            assert word.lower() not in firsts | lasts


# -- chunking ----------------------------------------------------------------------------------


def _para(n_words: int, tag: str) -> str:
    words = [f"{tag}{i}" for i in range(n_words - 1)]
    return " ".join(words) + " end."


def test_chunks_snap_to_paragraph_boundaries():
    text = "\n\n".join(_para(30, f"p{i}w") for i in range(12))  # 360 words
    doc = {"source": "fixture", "doc_id": "d", "text": text}
    chunks = chunk.chunk_doc(doc, TOK, sizes=[100], tolerance=0.1)
    assert [c["n_tokens"] for c in chunks] == [90, 90, 90, 90]
    for c in chunks:
        s, e = c["char_offsets"]
        assert text[s:e] == c["text"]
        assert c["boundary"] == "para"
        assert e == len(text) or text[e : e + 2] == "\n\n"
        assert c["sha256"] and c["split"] in ("train", "val", "test")
    assert chunks[0]["chunk_index"] == 0 and chunks[-1]["chunk_index"] == 3


def test_oversized_paragraph_falls_back_to_sentences():
    sentences = " ".join(f"Sentence {i} has exactly seven words here." for i in range(60))
    text = _para(20, "intro") + "\n\n" + sentences + "\n\n" + _para(20, "outro")
    doc = {"source": "fixture", "doc_id": "big", "text": text}
    chunks = chunk.chunk_doc(doc, TOK, sizes=[100], tolerance=0.1)
    assert all(c["n_tokens"] <= 110 for c in chunks)
    assert sum(c["n_tokens"] for c in chunks) == TOK.count(text)
    for c in chunks[:-1]:
        assert c["boundary"] in ("para", "sentence")
        assert c["text"].endswith(".")
    covered = " ".join(c["text"] for c in chunks).split()
    assert covered == text.split()  # nothing lost or duplicated


def test_chunk_size_is_seeded_per_document_and_sessions_stay_whole():
    doc = {
        "source": "fixture",
        "doc_id": "x",
        "text": "\n\n".join(_para(30, "a") for _ in range(20)),
    }
    sizes = {
        chunk.chunk_doc(doc, TOK, sizes=[64, 128, 256], seed=s)[0]["chunk_size"] for s in range(12)
    }
    assert len(sizes) > 1
    a = chunk.chunk_doc(doc, TOK, sizes=[64, 128, 256], seed=5)
    assert a == chunk.chunk_doc(doc, TOK, sizes=[64, 128, 256], seed=5)
    row = json.loads((CORPUS / "conversations.jsonl").read_text())
    spec = fetch.spec_for({"key": "conv_chronicles", "kind": "local", "path": "-"})
    conv = fetch.to_doc(row, spec, 0)
    assert conv["text"].count("[Session") == 5 and "[Session 2: A few days after]" in conv["text"]
    sessions = chunk.chunk_doc(conv, TOK, sizes=[1024], mode="session")
    assert len(sessions) == 5 and all(c["text"].startswith("[Session") for c in sessions)
    assert sessions[0]["chunk_size"] == "session"


def test_splits_are_by_document():
    assert chunk.assign_split("s", "doc") == chunk.assign_split("s", "doc")
    splits = [chunk.assign_split("s", str(i), val=0.2, test=0.2) for i in range(500)]
    assert 50 < splits.count("test") < 150 and 50 < splits.count("val") < 150


# -- episodes ----------------------------------------------------------------------------------


def _bio_chunks(seed=0, people=40, size=120):
    doc, probes = bios.generate_doc(seed, 0, people=people, p_update=0.6)
    return chunk.chunk_doc(doc, TOK, sizes=[size], tolerance=0.1), probes


def test_episode_structure():
    chunks, probes = _bio_chunks()
    assert len(chunks) > 12
    eps = episodes.build_episodes(chunks, probes, seed=1, k_max=4, p_mixed=0.0, max_probes=50)
    used = []
    for e in eps:
        idx = [int(w["chunk_id"].rsplit("/", 1)[1]) for w in e["writes"]]
        target = int(e["target"]["chunk_id"].rsplit("/", 1)[1])
        assert 1 <= e["k"] == len(idx) <= 4
        assert idx == list(range(idx[0], idx[0] + len(idx))) and target == idx[-1] + 1
        assert not e["mixed"] and len(e["doc_ids"]) == 1
        used += idx + [target]
        for p in e["probes"]:
            assert p["lag"] == e["k"] - 1 - p["write_index"]
            assert p["lag_bucket"] == episodes.lag_bucket(p["lag"])
            if p["type"] == "abstain":
                assert p["answer"] is None
            else:
                assert p["answer"] in p["evidence"]
                assert p["evidence"] in e["writes"][p["write_index"]]["text"]
    assert len(used) == len(set(used))  # each chunk in at most one episode


def test_update_probes_expect_the_latest_value_seen():
    chunks, probes = _bio_chunks(seed=2)
    eps = episodes.build_episodes(chunks, probes, seed=0, k_max=64, p_mixed=0.0, max_probes=10**6)
    by_id = {p["id"]: p for p in probes}
    n_updates = 0
    for e in eps:
        span = [w["char_offsets"] for w in e["writes"]]
        for p in e["probes"]:
            src_probe = by_id[p["id"]]
            seen = [
                f
                for f in src_probe["facts"]
                if any(s <= f["start"] and f["end"] <= t for s, t in span)
            ]
            if src_probe["facts"]:
                assert p["answer"] == seen[-1]["value"]
            if p["type"] == "update":
                n_updates += 1
                assert p["reject"] and p["answer"] not in p["reject"]
    assert n_updates > 0


def test_dropped_chunk_ends_the_run_and_mixed_episodes():
    chunks, probes = _bio_chunks()
    gap = [c for c in chunks if c["chunk_index"] != 5]
    eps = episodes.build_episodes(gap, probes, seed=0, k_max=8, p_mixed=0.0)
    for e in eps:
        idx = [int(w["chunk_id"].rsplit("/", 1)[1]) for w in e["writes"]]
        idx.append(int(e["target"]["chunk_id"].rsplit("/", 1)[1]))
        assert 5 not in idx and not (min(idx) < 5 < max(idx))
    other, more = _bio_chunks(seed=9)
    mixed = episodes.build_episodes(chunks + other, probes + more, seed=0, k_max=12, p_mixed=1.0)
    m = [e for e in mixed if e["mixed"]]
    assert m
    for e in m:
        assert len(e["doc_ids"]) >= 2 and e["k"] <= 12
        assert sum(w["boundary_before"] for w in e["writes"]) == len(e["doc_ids"]) - 1
    assert episodes.build_episodes(chunks, probes, seed=3) == episodes.build_episodes(
        chunks, probes, seed=3
    )


def test_select_probes_spreads_across_lag_buckets():
    import random

    ps = [
        {"id": str(i), "lag_bucket": b, "write_index": i}
        for i, b in enumerate(["0"] * 20 + ["1-3"] * 3 + ["16+"] * 2)
    ]
    chosen = episodes.select_probes(ps, 6, random.Random(0))
    assert len(chosen) == 6
    assert {p["lag_bucket"] for p in chosen} == {"0", "1-3", "16+"}


# -- decontamination ---------------------------------------------------------------------------


def _chunk(cid, text, source="fixture"):
    return {"id": cid, "source": source, "doc_id": cid, "text": text}


def test_reference_set_loads_what_exists(ref):
    s = {x["name"]: x for x in ref.summary()["sources"]}
    assert s[str(CORPUS / "absent.json")]["status"] == "absent"
    assert s[str(FIX / "longmemeval_tiny.json")]["status"] == "loaded"
    assert ref.summary()["templates"] > 30 and ref.summary()["canaries"] == 3


def test_decontam_catches_template_canary_and_ngram(ref, bench_users):
    canary = bench_users[0].canary
    name = bench_users[1].name
    clean = "The keeper climbed the stairs every evening to trim the wicks of the lamp."
    chunks = [
        _chunk("clean", clean),
        _chunk("template", clean + " Anyway, my dog Pollux keeps stealing socks. " + clean),
        _chunk("canary", f"{clean} The code painted on the crate read {canary} in red."),
        _chunk("name", f"{clean} A letter arrived for {name} this morning."),
        _chunk(
            "ngram",
            "He said: scraping the old varnish off his little wooden rowing boat "
            "before repainting the whole hull, which took weeks.",
        ),
        _chunk("gen", "My partner Zarnel thinks I work too late.", source="engram_bios"),
    ]
    kept, report = decontam.decontaminate_chunks(chunks, ref, generated_sources={"engram_bios"})
    assert [c["id"] for c in kept] == ["clean"]
    kinds = {ex["chunk"]: {h["kind"] for h in ex["hits"]} for ex in report["examples"]}
    assert kinds["template"] == {"template"}
    assert "canary" in kinds["canary"] and "name" in kinds["name"]
    assert kinds["ngram"] == {"ngram"}
    ngram_ref = next(h for ex in report["examples"] if ex["chunk"] == "ngram" for h in ex["hits"])
    assert ngram_ref["ref"].endswith("eval_tiny.json")
    fatal = {(f["chunk"], f["kind"]) for f in report["fatal"]}
    # canaries and user names are always fatal; a template is fatal only in generated data
    assert fatal == {("canary", "canary"), ("name", "name"), ("gen", "template")}


def test_bench_templates_compile_and_match_rendered_sessions(ref, bench_users):
    turns = [t.text for u in bench_users for s in u.sessions for t in s.turns if t.event]
    planted = [t for t in turns if "locker code" not in t]
    hits = [
        bool([h for h in ref.check_text(t, ngrams=False) if h["kind"] == "template"])
        for t in planted
    ]
    assert sum(hits) / len(hits) > 0.95


def test_probe_level_filter(ref):
    probes = [
        {"id": "a", "source": "engram_bios", "question": "What is my sister's name?", "facts": []},
        {
            "id": "b",
            "source": "x",
            "question": "What colour was the boat my uncle repainted last spring?",
            "facts": [],
        },
        {
            "id": "c",
            "source": "x",
            "question": "Which city was Halard Pernellvale born in?",
            "facts": [{"value": "Ostanport"}],
        },
    ]
    kept, rep = decontam.decontaminate_probes(probes, ref, generated_sources={"engram_bios"})
    assert [p["id"] for p in kept] == ["c"]
    assert rep["fatal"] and rep["fatal"][0]["probe"] == "a"


def test_bios_pass_decontam(ref):
    docs, probes = bios.generate(0, 2, people=30)
    chunks = [c for d in docs for c in chunk.chunk_doc(d, TOK, sizes=[200])]
    kept, report = decontam.decontaminate_chunks(chunks, ref, generated_sources={"engram_bios"})
    assert len(kept) == len(chunks) and not report["fatal"]
    kept_p, rep = decontam.decontaminate_probes(probes, ref, generated_sources={"engram_bios"})
    assert len(kept_p) == len(probes) and not rep["fatal"]


# -- card, probes interface, fetch guards ------------------------------------------------------


def test_card_stats():
    docs = [
        {"source": "gutenberg", "doc_id": "1", "text": "one two three four", "licence": "PD"},
        {"source": "gutenberg", "doc_id": "2", "text": "five six", "licence": "PD"},
        {"source": "arxiv", "doc_id": "3", "text": "a b c", "licence": "CC BY 4.0"},
    ]
    stats = card.doc_stats(docs, TOK)
    g = stats["sources"]["gutenberg"]
    assert (g["docs"], g["tokens"], g["mean_doc_tokens"]) == (2, 6, 3.0)
    assert g["licence"] == src.REGISTRY["gutenberg"].licence and g["slice"] == "A"
    assert stats["sources"]["arxiv"]["doc_licences"] == {"CC BY 4.0": 1}
    assert (stats["total_docs"], stats["total_tokens"], stats["approximate"]) == (3, 9, True)
    assert get_tokenizer("whitespace:1.5").count("a b c d") == 6


def test_registry_and_licences():
    assert src.REGISTRY["pg19"].licence == '"Apache License, Version 2.0"'
    assert src.REGISTRY["stackexchange"].verdict == "ablation"
    assert src.normalise_licence("CC-BY-SA-4.0") == "CC-BY-SA"
    assert src.normalise_licence("cc0-1.0") == "CC0"
    assert src.normalise_licence("CC BY-NC 4.0") is None
    assert src.normalise_licence("Public Domain") == "PD"
    assert src.normalise_licence("CCBY") == "CC-BY"  # peS2o's metadata.oa_license
    assert src.normalise_licence("CC BY-NC-SA 4.0") is None
    row = {"metadata": {"oa_license": "CCBY"}, "text": "x", "id": "1"}
    assert fetch.to_doc(row, fetch.spec_for({"key": "pes2o"}), 0)["licence"] == "CCBY"


class FakeChat:
    name = "fake-generator"

    def complete(self, messages, *, max_tokens=512):
        return json.dumps(
            [
                {
                    "question": "How many steps led to the lamp room?",
                    "answer": "one hundred and twelve",
                    "evidence_span": "one hundred and twelve steps",
                    "type": "fact",
                },
                {
                    "question": "Was the tower made of granite?",
                    "answer": "yes",
                    "evidence_span": "grey granite",
                    "type": "fact",
                },
                {
                    "question": "Who painted the tower?",
                    "answer": "the keeper",
                    "evidence_span": "not in the passage",
                    "type": "fact",
                },
                {
                    "question": "What was the keeper's dog called?",
                    "answer": "not stated",
                    "evidence_span": "",
                    "type": "abstain",
                },
            ]
        )


def test_llm_probe_interface(tmp_path):
    doc = next(d for d in read_jsonl(CORPUS / "docs.jsonl") if d["doc_id"] == "lighthouse")
    doc["source"] = "fixture"
    chunks = chunk.chunk_doc(doc, TOK, sizes=[80])
    dry = llm_probes.run_stage(chunks, tmp_path / "dry", generator=None, dry_run=True)
    assert dry["requests"] == len(chunks) and dry["dry_run"]
    req = next(read_jsonl(tmp_path / "dry" / "llm_requests.jsonl"))
    assert chunks[0]["text"] in req["messages"][0]["content"] and len(req["prompt_sha256"]) == 64

    gen = llm_probes.LLMProbeGenerator(FakeChat())
    out = llm_probes.run_stage(chunks[:2], tmp_path / "wet", generator=gen, dry_run=False)
    probes = list(read_jsonl(tmp_path / "wet" / "llm.jsonl"))
    assert out["dropped"]["yes_no"] == 2 and out["dropped"]["span_not_substring"] >= 1
    fact = next(p for p in probes if p["type"] == "fact")
    s, e = fact["facts"][0]["start"], fact["facts"][0]["end"]
    assert doc["text"][s:e] == "one hundred and twelve steps"
    assert any(p["type"] == "abstain" and p["anchor"] for p in probes)


def test_fetch_guards(tmp_path):
    with pytest.raises(ValueError, match="lineage-excluded"):
        fetch.check_allowed({"key": "x", "hf_id": "lmsys/lmsys-chat-1m"})
    with pytest.raises(ValueError, match="ablation"):
        fetch.check_allowed(fetch.spec_for({"key": "ubuntu_irc"}))
    # an uncapped network fetch is refused before anything is downloaded
    assert main(["fetch", "--source", "gutenberg", "--out", str(tmp_path)]) == 2
    n = main(
        [
            "fetch",
            "--config",
            str(_config(tmp_path, [])),
            "--source",
            "fixture_books",
            "--out",
            str(tmp_path / "f"),
            "--cap",
            "2",
        ]
    )
    assert n == 0 and len(list(read_jsonl(tmp_path / "f" / "raw" / "fixture_books.jsonl"))) == 2


# -- end to end --------------------------------------------------------------------------------


def _config(tmp_path: Path, extra_docs: list[dict]) -> Path:
    books = CORPUS / "docs.jsonl"
    if extra_docs:
        books = tmp_path / "books.jsonl"
        lines = (CORPUS / "docs.jsonl").read_text().splitlines()
        lines += [json.dumps(d) for d in extra_docs]
        books.write_text("\n".join(lines) + "\n")
    cfg = {
        "name": "fixture",
        "seed": 0,
        "target_tokens": 1e5,
        "tokenizer": "whitespace",
        "sources": [
            {
                "key": "fixture_books",
                "kind": "local",
                "path": str(books),
                "slice": "A",
                "licence": "fixture text (Apache-2.0)",
                "share": 0.5,
            },
            {
                "key": "conv_chronicles",
                "kind": "local",
                "path": str(CORPUS / "conversations.jsonl"),
            },
            {"key": "engram_bios", "share": 0.2},
        ],
        "filter": {"min_chunks": 3, "min_chunk_tokens": 40},
        "chunk": {"sizes": [40, 80], "tolerance": 0.1, "val": 0.0, "test": 0.0},
        "bios": {"docs": 2, "people_per_doc": 12},
        "decontam": {
            "bench": BENCH,
            "external": [str(CORPUS / "eval_tiny.json"), str(FIX / "locomo_tiny.json")],
        },
        "episodes": {"k_max": 4, "p_mixed": 0.3, "max_probes": 8},
        "pack": {"shard_size": 3},
        "probes": {"generator": {"model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506"}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_build_end_to_end(tmp_path):
    planted = {
        "doc_id": "planted",
        "text": "\n\n".join(
            ["The ferry crossed the bay twice a day in calm weather, carrying mail and passengers."]
            * 6
            + ["Then she wrote: I do everything in Helix, so keybinding tips should assume that."]
            + ["The harbour master logged each crossing in a book kept beside the ticket window."]
            * 6
        ),
    }
    config = _config(tmp_path, [planted])
    out = tmp_path / "out"
    rc = main(["build", "--config", str(config), "--out", str(out), "--dry-run"])
    assert rc == 0
    for f in [
        "fetch_manifest.json",
        "filter_report.json",
        "chunks.jsonl",
        "episodes.jsonl",
        "contamination_report.json",
        "card.json",
        "DATA_CARD.md",
        "manifest.json",
        "probes/llm_requests.jsonl",
        "decontam/probes.jsonl",
    ]:
        assert (out / f).exists(), f
    report = json.loads((out / "contamination_report.json").read_text())
    assert report["chunks"]["fixture_books"]["by_kind"].get("template", 0) >= 1
    assert not report["fatal"]
    eps = list(read_jsonl(out / "episodes.jsonl"))
    assert eps and {"fixture_books", "conv_chronicles", "engram_bios"} <= {
        s for e in eps for s in e["sources"]
    }
    assert {"fact", "abstain"} <= {p["type"] for e in eps for p in e["probes"]}
    kept = {c["id"] for c in read_jsonl(out / "decontam" / "chunks.jsonl")}
    assert all(w["chunk_id"] in kept for e in eps for w in e["writes"])
    c = json.loads((out / "card.json").read_text())
    assert c["episodes"]["episodes"] == len(eps)
    assert c["chunks"]["engram_bios"]["docs"] == 2
    assert c["docs"]["sources"]["conv_chronicles"]["licence"] == "cc-by-4.0"
    shards = sorted((out / "shards").glob("*.jsonl"))
    assert sum(len(list(read_jsonl(s))) for s in shards) == len(eps)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 0 and set(manifest["stage_seconds"]) >= {"fetch", "episodes"}
    requests = list(read_jsonl(out / "probes" / "llm_requests.jsonl"))
    assert requests and not any(r["chunk_id"].startswith("engram_bios") for r in requests)
    summary = json.loads((out / "probes" / "llm_summary.json").read_text())
    assert summary["generator"] == "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    again = tmp_path / "again"
    assert main(["build", "--config", str(config), "--out", str(again), "--dry-run"]) == 0
    assert (again / "episodes.jsonl").read_text() == (out / "episodes.jsonl").read_text()
    assert (
        main(
            [
                "build",
                "--config",
                str(config),
                "--out",
                str(again),
                "--resume",
                "--dry-run",
                "--stages",
                "episodes,card",
            ]
        )
        == 0
    )


def test_build_fails_on_planted_canary(tmp_path, bench_users):
    canary = bench_users[2].canary
    planted = {
        "doc_id": "leak",
        "text": "\n\n".join(
            [
                f"Paragraph {i} about the lock keepers and their long registers of boats."
                for i in range(14)
            ]
            + [f"The crate was stamped {canary} before it left the yard."]
        ),
    }
    config = _config(tmp_path, [planted])
    with pytest.raises(BuildError, match="fatal"):
        ctx = Context(config=json.loads(config.read_text()), out=tmp_path / "o", dry_run=True)
        run(ctx, parse_stages("fetch,filter,bios,chunk,decontam"), log=lambda m: None)
    assert main(["build", "--config", str(config), "--out", str(tmp_path / "o2"), "--dry-run"]) == 1


def test_document_filters():
    from engram.corpus import filter as flt

    body = "\n\n".join(_para(60, f"the{i} of and") for i in range(8))
    specs = {
        "pg19": fetch.spec_for({"key": "pg19"}),
        "gutenberg": fetch.spec_for({"key": "gutenberg"}),
        "wikipedia": fetch.spec_for({"key": "wikipedia"}),
        "pes2o": fetch.spec_for({"key": "pes2o"}),
        "arxiv": fetch.spec_for({"key": "arxiv"}),
    }
    url = "http://www.gutenberg.org/ebooks/1022"
    docs = {
        "pg19": [
            {
                "source": "pg19",
                "doc_id": "1022",
                "text": body,
                "force_split": "val",
                "meta": {"url": url},
            }
        ],
        "gutenberg": [
            {"source": "gutenberg", "doc_id": "1022", "text": "other " + body, "meta": {}},
            {"source": "gutenberg", "doc_id": "7", "text": body.replace("the0", "a0"), "meta": {}},
            {"source": "gutenberg", "doc_id": "8", "text": "Bonjour " * 400, "meta": {}},
        ],
        "wikipedia": [
            {
                "source": "wikipedia",
                "doc_id": "1",
                "meta": {},
                "text": "Ada Quell (born 4 May 1950) is a painter.\n\n" + body.replace("the", "x"),
            },
        ],
        "pes2o": [
            {
                "source": "pes2o",
                "doc_id": "2",
                "text": "zz " + body.replace("of", "in"),
                "licence": "CC BY-NC",
                "meta": {},
            }
        ],
        "arxiv": [
            {
                "source": "arxiv",
                "doc_id": "2402.17753",
                "text": body[::-1],
                "licence": "CC0",
                "meta": {},
            }
        ],
    }
    kept, report = flt.filter_docs(docs, specs, {"min_chunk_tokens": 40}, TOK)
    dropped = {n: r["dropped"] for n, r in report["sources"].items()}
    assert report["holdout_gutenberg_ids"] == ["1022"]
    assert dropped["gutenberg"] == {"pg19_holdout": 1, "near_duplicate": 1, "language": 1}
    assert dropped["wikipedia"] == {"biography": 1}
    assert dropped["pes2o"] == {"licence": 1}
    assert dropped["arxiv"] == {"arxiv_denylist": 1}
    assert [d["doc_id"] for d in kept["pg19"]] == ["1022"]


def test_v0_config_mirrors_the_plan():
    cfg = json.loads((Path(__file__).parents[1] / "configs" / "corpus-v0.json").read_text())
    ctx = Context(config=cfg, out=Path("unused"))
    keys = [s["key"] for s in ctx.specs()]
    assert keys.count("pg19") == 2 and "engram_threads" not in keys
    assert all(k in src.REGISTRY for k in keys)
    assert all(s["revision"] for s in ctx.specs() if s.get("hf_id"))
    shares = sum(e.get("share") or 0 for e in cfg["sources"])
    assert 0.95 <= shares <= 1.0
    assert parse_stages("generate,probe,card") == ["bios", "probes", "card"]
