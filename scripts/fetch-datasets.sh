#!/usr/bin/env bash
# Download the public benchmarks engram-bench reuses into data/external/ (gitignored) and
# convert them. LongMemEval _m (2.7 GB) is skipped; pass --with-m to fetch it too.
set -euo pipefail
mkdir -p data/external
hf=https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main
files=(longmemeval_oracle.json longmemeval_s_cleaned.json)
[[ ${1:-} == --with-m ]] && files+=(longmemeval_m_cleaned.json)
for f in "${files[@]}"; do
  [[ -s data/external/$f ]] || curl -fL --retry 3 -o "data/external/$f" "$hf/$f"
done
[[ -s data/external/locomo10.json ]] || curl -fL --retry 3 -o data/external/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
uv run engram-bench import longmemeval data/external/longmemeval_s_cleaned.json --out data/longmemeval-s
uv run engram-bench import longmemeval data/external/longmemeval_oracle.json --out data/longmemeval-oracle
uv run engram-bench import locomo data/external/locomo10.json --out data/locomo
