from engram.cli import main as bench_main
from engram.compute import frozen_base_training_flops, gpu_hours
from engram.compute import main as compute_main


def test_cli_end_to_end(tmp_path, capsys):
    data, run_dir, floor_dir = tmp_path / "bench", tmp_path / "run", tmp_path / "floor"
    bench_main(["generate", "--out", str(data), "--users", "3", "--sessions", "9", "--seed", "2"])
    bench_main(["run", "--bench", str(data), "--system", "oracle", "--out", str(run_dir)])
    bench_main(["run", "--bench", str(data), "--system", "b0", "--mock", "null",
                "--out", str(floor_dir), "--no-leakage"])  # fmt: skip
    bench_main(["report", str(run_dir), "--floor", str(floor_dir)])
    out = capsys.readouterr().out
    assert "ALL" in out and "oracle" in out
    assert (run_dir / "manifest.json").exists() and (run_dir / "leakage.jsonl").exists()


def test_cli_llm_writer_with_mock_falls_back(tmp_path, capsys):
    # The null model drops every value, so every rewrite must fall back to the template.
    bench_main(["generate", "--out", str(tmp_path), "--users", "2", "--sessions", "6",
                "--writer", "llm", "--writer-mock", "null"])  # fmt: skip
    assert "fell back" in capsys.readouterr().out


def test_compute_matches_experiment_plan(capsys):
    # experiment-plan.md: 4 × 8e9 × 1e10 ≈ 3.2e20 FLOPs ≈ 220 GPU-hours at 400 TFLOP/s.
    flops = frozen_base_training_flops(8e9, 1e10)
    assert abs(flops - 3.2e20) < 1e18
    assert 215 < gpu_hours(flops, 400) < 230
    compute_main([])
    assert "GPU-hours" in capsys.readouterr().out


def test_cli_import_and_judge(tmp_path, capsys):
    bench, run_dir = tmp_path / "locomo", tmp_path / "run"
    bench_main(["import", "locomo", "tests/fixtures/locomo_tiny.json", "--out", str(bench)])
    bench_main(["run", "--bench", str(bench), "--system", "oracle", "--no-leakage",
                "--out", str(run_dir)])  # fmt: skip
    # A judge that never says PASS must flip every judged probe to wrong, and only those.
    bench_main(["judge", str(run_dir), "--bench", str(bench), "--model", "x", "--mock", "null"])
    import json

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["judge"]["judged"] > 0
    rows = [json.loads(x) for x in (run_dir / "results.jsonl").read_text().splitlines()]
    assert sum(not r["correct"] for r in rows) == manifest["judge"]["judged"]
    assert manifest["judge"]["failed"] == 0


def test_cli_calibrate_roundtrip(tmp_path, capsys):
    import csv

    data, run_dir, sheet = tmp_path / "b", tmp_path / "r", tmp_path / "labels.csv"
    bench_main(["generate", "--out", str(data), "--users", "2", "--sessions", "6", "--seed", "3"])
    bench_main(["run", "--bench", str(data), "--system", "oracle", "--no-leakage",
                "--out", str(run_dir)])  # fmt: skip
    bench_main(["calibrate", "export", str(sheet), "--run", str(run_dir), "--bench", str(data),
                "--n", "5"])  # fmt: skip
    with open(sheet, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    for r in rows:
        r["human"] = "1"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bench_main(["calibrate", "score", str(sheet), "--mock", "null"])
    assert "agreement=0.000" in capsys.readouterr().out
