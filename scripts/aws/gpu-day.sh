#!/usr/bin/env bash
# Phase 0 GPU work, run ON the instance (via `gpu.sh exec 'scripts/aws/gpu-day.sh'`), after
# `gpu.sh wait` reports vLLM serving the base model. Every step is resumable: model responses
# are cached, so re-running after a spot interruption only redoes missing calls.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-8B}
WORKERS=${WORKERS:-16}
tag=${MODEL//\//-}
out=runs/gpu-day
mkdir -p "$out"
run() { uv run engram-bench run --model "$MODEL" --workers "$WORKERS" --no-session-replies "$@"; }

# 0. Datasets: engram-bench v0 is generated from its seed (same hash as anywhere else).
[[ -d data/bench-v0 ]] || uv run engram-bench generate --out data/bench-v0 --users 50 --sessions 30 --seed 0
[[ -d data/bench-pilot ]] || uv run engram-bench generate --out data/bench-pilot --users 10 --sessions 30 --seed 0
scripts/fetch-datasets.sh

# 1. Serving throughput (issue #12).
uv run engram-measure serve --model "$MODEL" --concurrency "$WORKERS" > "$out/measure-serve.json"

# 2. Base-model baselines on engram-bench v0 (issue #3).
run --bench data/bench-v0 --system b0 --no-leakage --out "$out/b0-$tag"
run --bench data/bench-v0 --system b1 --leak-targets 2 --out "$out/b1-$tag"
run --bench data/bench-v0 --system b3 --retriever hybrid --leak-targets 2 --out "$out/b3-$tag"

# 3. B3 on the reused benchmarks (no canaries there, so no leakage pass).
run --bench data/longmemeval-s --system b3 --retriever hybrid --no-leakage --out "$out/b3-lme-s-$tag"
run --bench data/locomo --system b3 --retriever hybrid --no-leakage --out "$out/b3-locomo-$tag"

# 4. B4 per-user LoRA needs the GPU to itself: stop vLLM first.
sudo systemctl stop vllm
uv run engram-measure sleep --base-model "$MODEL" --messages 150 > "$out/measure-sleep.json"
uv run engram-bench run --bench data/bench-pilot --system b4 --base-model "$MODEL" \
  --no-session-replies --leak-targets 1 --out "$out/b4-pilot-$tag"
sudo systemctl start vllm
echo "done; now run: scripts/aws/gpu.sh push-runs (from your machine)"
