#!/usr/bin/env bash
# Phase 0 frontier pilot on subscription models: B0, B1, B2 for one provider.
#   scripts/pilot.sh claude:claude-opus-5-5 data/bench-pilot runs/pilot
# Responses are cached, so re-running after an interruption only makes the missing calls.
set -euo pipefail
model=$1 bench=$2 out=$3 workers=${WORKERS:-4}
tag=${model//[:\/]/-}
common=(--bench "$bench" --model "$model" --no-session-replies --workers "$workers")
uv run engram-bench run --system b0 "${common[@]}" --no-leakage --out "$out/b0-$tag" -v
uv run engram-bench run --system b1 "${common[@]}" --leak-targets 1 --out "$out/b1-$tag" -v
uv run engram-bench run --system b2 "${common[@]}" --retriever hybrid --leak-targets 1 \
  --out "$out/b2-$tag" -v
