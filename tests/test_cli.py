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
