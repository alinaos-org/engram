import hashlib
import json
import platform
import re
import shutil
import stat
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from engram.bench import dataset, seal
from engram.bench.runner import _git_commit, apply_floor, load_run, save_run, summary_json
from engram.bench.scoring import summarise
from engram.cli import main as bench_main
from engram.models.subscription import ClaudeCLI

TINY = ["--users", "3", "--sessions", "9"]
FIELDS = {"scheme", "seed", "salt", "spec_sha256", "seed_commitment"}


@pytest.fixture(autouse=True)
def _outside_repos(tmp_path):
    if seal.work_tree(tmp_path) is not None:
        pytest.skip("the pytest tmp dir is inside a git work tree")


def _sealed(tmp_path: Path, name: str = "b", *extra: str) -> tuple[Path, Path]:
    """Generate a tiny sealed split; returns its directory and its seed file."""
    out, seed_file = tmp_path / name, tmp_path / f"{name}.seed.json"
    bench_main(["generate", "--seal", str(seed_file), "--out", str(out), *TINY, *extra])
    return out, seed_file


def _repo(tmp_path: Path, git_file: bool = False) -> Path:
    """A fake work tree: ``.git`` as a directory, or as the file worktrees use."""
    repo = tmp_path / "repo"
    repo.mkdir()
    if git_file:
        (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/repo\n")
    else:
        (repo / ".git").mkdir()
    return repo


def _found(root: Path, needles: list[str]) -> list[str]:
    return [
        f"{f}: {n}" for f in root.rglob("*") if f.is_file() for n in needles if n in f.read_text()
    ]


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text())


def test_commitment_known_answer():
    spec = seal.spec_sha256(asdict(dataset.BenchConfig()), "template")
    assert spec == "1a301dc710f5d2f013b3a53411a3304f4312bbf64abbb8f6d65f66afb8d8a26d"
    commitment = seal.commitment("11" * 32, "22" * 32, spec)
    assert commitment == "82dc41127dcaa0e4a7aa6a1ed76c014c27edc176b2b3b95510ff7a92932268e1"
    # The seed key is dropped, not nulled, so the spec hash ignores it.
    assert seal.spec_sha256({**asdict(dataset.BenchConfig()), "seed": None}, "template") == spec
    a, b = seal.new_record(spec), seal.new_record(spec)
    assert a["seed"] != b["seed"] and a["salt"] != b["salt"]
    seal.verify(seal.sealed_block(a), a)


def test_sealed_generate(tmp_path, capsys):
    out, seed_file = _sealed(tmp_path)
    printed = capsys.readouterr().out
    manifest, record = _manifest(out), json.loads(seed_file.read_text())
    assert "seed" in manifest["config"] and manifest["config"]["seed"] is None
    assert list(manifest)[-1] == "sealed"
    assert list(manifest["sealed"]) == [
        "scheme", "spec_sha256", "seed_commitment", "git_commit", "python"
    ]  # fmt: skip
    assert manifest["sealed"]["python"] == platform.python_version()
    assert set(record) == FIELDS and seal.read_seed_file(seed_file) == record
    assert stat.S_IMODE(seed_file.stat().st_mode) == 0o600
    assert manifest["sealed"].items() >= seal.sealed_block(record).items()
    assert record["seed_commitment"] in printed and record["seed"] not in printed
    assert not _found(out, [record["seed"], str(seal.seed_int(record))])
    # The normative mapping (sealed-split.md), written out rather than through seal.seed_int.
    config = dataset.BenchConfig(**{**manifest["config"], "seed": int(record["seed"], 16)})
    assert dataset.content_hash(dataset.generate(config)) == manifest["sha256"]
    assert dataset.load(out).config.seed is None
    assert dataset.load(out, seed_file).sealed == manifest["sealed"]


@pytest.mark.parametrize("git_file", [False, True])
def test_seed_file_refused(tmp_path, git_file):
    repo, out = _repo(tmp_path, git_file), tmp_path / "b"
    existing = tmp_path / "old.seed.json"
    existing.write_text("original\n")
    (tmp_path / "link").symlink_to(repo)
    (tmp_path / "file-link.seed.json").symlink_to(repo / "y.seed.json")
    cases = [
        (repo / "x.seed.json", "work tree"),
        (repo / "sub" / "x.seed.json", "work tree"),
        (tmp_path / "link" / "x.seed.json", "work tree"),
        (tmp_path / "file-link.seed.json", "work tree"),
        (out / "x.seed.json", "dataset directory"),
        (existing, "never overwritten"),
    ]
    for seed_file, why in cases:
        with pytest.raises(ValueError, match=why):  # --allow-in-repo never covers the seed file
            bench_main(["generate", "--seal", str(seed_file), "--out", str(out), *TINY,
                        "--allow-in-repo"])  # fmt: skip
        assert not out.exists()
    assert existing.read_text() == "original\n"
    assert not (repo / "sub").exists() and not list(repo.rglob("*.seed.json"))


def test_sealed_out_in_repo_needs_override(tmp_path):
    repo, seed_file = _repo(tmp_path), tmp_path / "s.seed.json"
    out = repo / "data" / "b"
    args = ["generate", "--seal", str(seed_file), "--out", str(out), *TINY]
    with pytest.raises(ValueError, match="--allow-in-repo"):
        bench_main(args)
    assert not seed_file.exists() and not out.exists()  # refused before the seed is drawn
    bench_main([*args, "--allow-in-repo"])
    assert dataset.load(out, seed_file).sealed
    # Unsealed datasets are never checked.
    plain, run_dir = repo / "data" / "plain", repo / "runs" / "oracle"
    bench_main(["generate", "--out", str(plain), *TINY])
    bench_main(["run", "--bench", str(plain), "--system", "oracle", "--out", str(run_dir)])
    assert (run_dir / "results.jsonl").exists()


def test_tilde_paths_checked_where_written(tmp_path, monkeypatch):
    # Shells leave "--out=~/x" as typed. The CLI expands it, so the guard and the writer see the
    # same path; a library caller's literal "~/x" is checked where it is written, under the cwd.
    repo, home = _repo(tmp_path), tmp_path / "home"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HOME", str(home))
    bench_main(["generate", "--seal=~/s.seed.json", "--out=~/b", *TINY])
    bench_main(["run", "--bench=~/b", "--system", "oracle", "--out=~/run"])
    assert dataset.load(home / "b", home / "s.seed.json").sealed
    assert (home / "run" / "results.jsonl").exists()
    with pytest.raises(ValueError, match="dataset directory"):
        bench_main(["generate", "--seal=~/b2/s.seed.json", "--out=~/b2", *TINY])
    with pytest.raises(ValueError, match="work tree"):
        save_run(load_run(home / "run"), Path("~/run"))
    assert not (repo / "~").exists()


def test_seal_flag_errors(tmp_path):
    seed_file, out = tmp_path / "x.seed.json", tmp_path / "b"
    # An empty --seal must not fall back to an unsealed split with the public default seed.
    # "--seed 0" equals the old default, which Python 3.11's exclusive group did not catch.
    # "~nosuchuser/..." cannot be expanded: a usage error, not a traceback.
    for flags in (
        ["--seed", "5", "--seal", str(seed_file)],
        ["--seed", "0", "--seal", str(seed_file)],
        ["--seal", ""],
        ["--seal", " "],
        ["--seal", "~engram-no-such-user-zz/x.seed.json"],
    ):
        with pytest.raises(SystemExit):
            bench_main(["generate", *flags, "--out", str(out), *TINY])
    assert not seed_file.exists() and not out.exists()


def test_sealed_split_never_overwritten(tmp_path):
    out, _ = _sealed(tmp_path)
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    new_seed = tmp_path / "new.seed.json"
    for flags in (["--seal", str(new_seed)], ["--seed", "1"], []):
        with pytest.raises(ValueError, match="never overwrit"):
            bench_main(["generate", *flags, "--out", str(out), *TINY])
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before
    assert not new_seed.exists()  # refused before the seed is drawn
    # --seal refuses any existing dataset; plain generate still rewrites an unsealed one.
    plain = tmp_path / "plain"
    bench_main(["generate", "--out", str(plain), *TINY])
    with pytest.raises(ValueError, match="never overwrit"):
        bench_main(["generate", "--seal", str(new_seed), "--out", str(plain), *TINY])
    assert not new_seed.exists()
    bench_main(["generate", "--out", str(plain), *TINY, "--seed", "3"])
    assert _manifest(plain)["config"]["seed"] == 3


def test_git_commit_is_the_engram_checkouts(tmp_path, monkeypatch):
    root = Path(seal.__file__).resolve().parents[2]

    def git(*args: str) -> str:
        cmd = ["git", "-C", str(root), *args]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    if shutil.which("git") is None or git("rev-parse", "--show-toplevel") != str(root):
        pytest.skip("engram is not running from its own git checkout")
    monkeypatch.chdir(tmp_path)  # outside every repository: the cwd is not what is recorded
    head = git("rev-parse", "HEAD")
    # GIT_DIR (set in hooks, say) overrides -C, so the recorded commit must not follow it.
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    assert _git_commit() in (head, f"{head}-dirty")


def test_load_rejects(tmp_path):
    out, seed_file = _sealed(tmp_path)
    _, other_seed = _sealed(tmp_path, "other")
    record = json.loads(seed_file.read_text())

    def load_edited(edit):
        copy = tmp_path / "edited"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(out, copy)
        manifest = _manifest(copy)
        edit(manifest)
        (copy / "manifest.json").write_text(json.dumps(manifest))
        return dataset.load(copy)

    def seed_file_with(**changes) -> Path:
        path = tmp_path / "changed.seed.json"
        path.write_text(json.dumps({**record, **changes}))
        return path

    with pytest.raises(ValueError, match="has a seed"):
        load_edited(lambda m: m["config"].update(seed=5))
    with pytest.raises(ValueError, match="spec_sha256"):
        load_edited(lambda m: m["config"].update(sleep_every=4))
    with pytest.raises(ValueError, match="unknown scheme"):
        load_edited(lambda m: m["sealed"].update(scheme="engram-seal-v0"))
    with pytest.raises(ValueError, match="different split"):
        dataset.load(out, other_seed)
    for name in ("seed", "salt"):
        flipped = format(int(record[name], 16) ^ 1, "064x")
        with pytest.raises(ValueError, match="do not open the commitment"):
            dataset.load(out, seed_file_with(**{name: flipped}))
    for changes in ({"seed": record["seed"].upper()}, {"note": "x"}, {"scheme": "engram-seal-v0"}):
        with pytest.raises(ValueError, match="not an engram seed file"):
            dataset.load(out, seed_file_with(**changes))
    bench_main(["generate", "--out", str(tmp_path / "plain"), *TINY])
    with pytest.raises(ValueError, match="not a sealed dataset"):
        dataset.load(tmp_path / "plain", seed_file)
    with pytest.raises(ValueError, match="seed is None"):
        dataset.generate(dataset.BenchConfig(users=1, sessions=6, seed=None))


def test_sealed_run_manifest_has_no_seed(tmp_path):
    out, seed_file = _sealed(tmp_path)
    run_dir = tmp_path / "run"
    bench_main(["run", "--bench", str(out), "--system", "oracle", "--out", str(run_dir)])
    manifest, bench_manifest = _manifest(run_dir), _manifest(out)
    assert manifest["bench"]["config"]["seed"] is None
    assert manifest["bench"]["sealed"] == bench_manifest["sealed"]
    assert manifest["bench_sha256"] == bench_manifest["sha256"]
    record = seal.read_seed_file(seed_file)
    assert not _found(run_dir, [record["seed"], str(seal.seed_int(record))])


def test_sealed_outputs_refused_in_repo(tmp_path, monkeypatch):
    # A claude: model whose calls are recorded (no CLI runs), so the run and judge refusals are
    # shown to come before the first model call, not only from save_run after it.
    calls = []

    def call(self, system, prompt):
        calls.append(prompt)
        return "PASS"

    monkeypatch.setattr("engram.models.cache.CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(ClaudeCLI, "_call", call)
    out, _ = _sealed(tmp_path)
    repo = _repo(tmp_path)
    run_dir, results = repo / "runs" / "x", repo / "runs" / "x" / "results.jsonl"
    args = ["run", "--bench", str(out), "--out", str(run_dir)]
    with pytest.raises(ValueError, match="--allow-in-repo"):
        bench_main([*args, "--system", "b0", "--model", "claude:x"])
    assert not run_dir.exists() and not calls
    bench_main([*args, "--system", "oracle", "--allow-in-repo"])
    # Flag every answer, so a judge that got past its check would call its model.
    rows = [json.loads(x) for x in results.read_text().splitlines()]
    results.write_text("".join(json.dumps({**r, "ambiguous": True}) + "\n" for r in rows))
    before = results.read_text()
    for cmd in (
        ["rescore", str(run_dir), "--bench", str(out)],
        ["judge", str(run_dir), "--bench", str(out), "--model", "claude:x"],
        ["calibrate", "export", str(repo / "l.csv"), "--run", str(run_dir), "--bench", str(out)],
    ):
        with pytest.raises(ValueError, match="work tree"):
            bench_main(cmd)
    assert results.read_text() == before and not (repo / "l.csv").exists() and not calls
    # The guards read the bench too, so a run manifest without the sealed block (as engram wrote
    # before sealed splits) is refused as well, and judge and rescore put the block back.
    manifest = _manifest(run_dir)
    del manifest["bench"]["sealed"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="work tree"):
        bench_main(["judge", str(run_dir), "--bench", str(out), "--model", "claude:x"])
    assert results.read_text() == before and not calls
    # The judge's check follows --out, and the flag lets in-repo writes through.
    bench_main(["judge", str(run_dir), "--bench", str(out), "--model", "x", "--mock", "null",
                "--out", str(tmp_path / "judged")])  # fmt: skip
    assert _manifest(tmp_path / "judged")["bench"]["sealed"] == _manifest(out)["sealed"]
    bench_main(["rescore", str(run_dir), "--bench", str(out), "--allow-in-repo"])
    result = load_run(run_dir)
    with pytest.raises(ValueError, match="work tree"):
        save_run(result, repo / "runs" / "y")
    assert not (repo / "runs" / "y").exists()
    save_run(result, repo / "runs" / "y", allow_in_repo=True)


def test_report_json_aggregate_only(tmp_path, capsys):
    out, _ = _sealed(tmp_path)
    run_dir, path = tmp_path / "run", tmp_path / "summary.json"
    bench_main(["run", "--bench", str(out), "--system", "oracle", "--out", str(run_dir)])
    capsys.readouterr()
    # The oracle is its own floor: it solves the gold probes, so the floor drops some.
    bench_main(["report", str(run_dir), "--floor", str(run_dir)])
    printed = capsys.readouterr().out
    bench_main(["report", str(run_dir), "--floor", str(run_dir), "--json", str(path)])
    assert capsys.readouterr().out == printed
    summary, text = json.loads(path.read_text()), path.read_text()
    assert list(summary) == ["manifest", "floor", "by_category", "by_timing", "leakage"]
    result = load_run(run_dir)
    assert summary["manifest"] == result.manifest
    no_gold = {"abstention", "forget", "incognito"}
    solved = sum(s.correct and s.category not in no_gold for s in result.scores)
    assert solved > 0 and summary["floor"] == {
        "system_name": result.manifest["system_name"],
        "bench_sha256": result.manifest["bench_sha256"],
        "dropped": solved,
    }
    assert {s["category"] for s in summary["by_category"]} >= {"ALL", "abstention"}
    floored = apply_floor(result.scores, result.scores)
    for key, by in (("by_category", "category"), ("by_timing", "timing")):
        expected = [asdict(s) for s in summarise(floored, by=by)]
        assert summary[key] == json.loads(json.dumps(expected))
    assert summary["leakage"] and all(v["checks"] > 0 for v in summary["leakage"].values())
    assert not re.search(r"u\d{4}-p\d+", text) and not re.search(r"\bu\d{4}\b", text)
    bench = dataset.load(out)
    assert not [u.canary for u in bench.users if u.canary in text]
    answers = [s.answer for s in result.scores] + [c.answer for c in result.leaks]
    assert not [a for a in answers if json.dumps(a) in text]
    assert not [p.question for p in bench.probes() if p.question in text]


def test_verify_seal_template(tmp_path, capsys):
    out, seed_file = _sealed(tmp_path)
    manifest = _manifest(out)
    counts = f"regenerated counts: ok (3 users, {manifest['n_probes']} probes)"
    capsys.readouterr()
    bench_main(["verify-seal", str(out), "--seed-file", str(seed_file)])
    assert capsys.readouterr().out.splitlines() == [
        "commitment: ok", "content sha256: ok", counts, "regenerated content: ok"
    ]  # fmt: skip
    bare = tmp_path / "test-manifest.json"
    shutil.copy(out / "manifest.json", bare)
    lines = dataset.verify_seal(bare, seed_file)
    assert lines == ["commitment: ok", counts, "regenerated content: ok"]
    _, other_seed = _sealed(tmp_path, "other")
    with pytest.raises(ValueError, match="different split"):
        dataset.verify_seal(out, other_seed)
    users = out / "users.jsonl"
    name = dataset.load(out).users[0].name
    users.write_text(users.read_text().replace(name, "Someone Else"))
    with pytest.raises(ValueError, match="hash"):
        dataset.verify_seal(out, seed_file)
    # With the hash updated to match, the regeneration catches the edit, and the checks that
    # passed before it are still printed.
    manifest["sha256"] = hashlib.sha256(users.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    capsys.readouterr()
    with pytest.raises(ValueError, match="contents differ.*Python"):
        bench_main(["verify-seal", str(out), "--seed-file", str(seed_file)])
    assert capsys.readouterr().out.splitlines() == ["commitment: ok", "content sha256: ok", counts]


@pytest.mark.parametrize("writer", ["long", "llm"])
def test_verify_seal_long_writer_plan(tmp_path, monkeypatch, writer):
    # A fake claude: writer whose session text differs from what the offline regeneration
    # writes, so the plan check is shown to ignore the sessions. No CLI runs.
    def call(self, system, prompt):
        if "\n\nMessage: " in prompt:  # LLMWriter: keep the planted strings, change the text
            return prompt.rsplit("\n\nMessage: ", 1)[1] + " (rewritten)"
        return json.dumps({"messages": [f"rewritten {i}" for i in range(3)]})  # LongWriter

    monkeypatch.setattr("engram.models.cache.CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(ClaudeCLI, "_call", call)
    out, seed_file = _sealed(
        tmp_path, "b", "--writer", writer, "--writer-model", "claude:x", "--no-cache"
    )
    assert "rewritten" in (out / "users.jsonl").read_text()
    assert dataset.verify_seal(out, seed_file)[-1] == "regenerated plan: ok"
    bare = tmp_path / "test-manifest.json"
    shutil.copy(out / "manifest.json", bare)
    lines = dataset.verify_seal(bare, seed_file)
    assert lines[0] == "commitment: ok"
    assert lines[-1] == "regenerated plan: not checked (no users.jsonl)"
    # Change one gold answer and update the content hash to match: only the plan check fails.
    users = out / "users.jsonl"
    rows = [json.loads(x) for x in users.read_text().split("\n") if x.strip()]
    probe = next(p for p in rows[0]["probes"] if p["gold"])
    probe["gold"] += " Jr"
    users.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = _manifest(out)
    manifest["sha256"] = hashlib.sha256(users.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert dataset.load(out, seed_file).sealed  # content hash and commitment still pass
    with pytest.raises(ValueError, match=f"plans for {rows[0]['id']} differ"):
        dataset.verify_seal(out, seed_file)


def test_unsealed_manifests_unchanged(tmp_path):
    data, run_dir = tmp_path / "b", tmp_path / "run"
    bench_main(["generate", "--out", str(data), *TINY, "--seed", "2"])
    manifest = _manifest(data)
    assert list(manifest) == ["engram_version", "config", "writer", "sha256", "n_users", "n_probes"]
    assert manifest["config"]["seed"] == 2 and dataset.load(data).sealed is None
    bench_main(["run", "--bench", str(data), "--system", "oracle", "--out", str(run_dir)])
    assert _manifest(run_dir)["bench"] == {"config": manifest["config"], "writer": "template"}
    # The published pilot still loads: its config and every run's manifest and results.
    pilot = Path(__file__).resolve().parent.parent / "results" / "pilot-2026-10-15"
    config = json.loads((pilot / "bench-manifest.json").read_text())["config"]
    assert dataset.BenchConfig(**config).seed == 0
    runs = [load_run(p) for p in sorted(pilot.iterdir()) if p.is_dir()]
    assert runs and not [r for r in runs if "sealed" in r.manifest["bench"]]
    assert summary_json(runs[0].manifest, runs[0].scores, runs[0].leaks)["by_category"]
