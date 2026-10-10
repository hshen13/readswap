#!/usr/bin/env bash
# Exploratory token sweep across output-row similarity (code/token_sweep.py), MATH500 seed 0.
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
for m in OctoThinker-3B-Short-LaSeR Qwen2.5-7B-LaSeR ORZ-7B-LaSeR; do
  [ -s "results/analysis/token_sweep_${m}.json" ] && continue
  $GPU_RUN env ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET=1 PYTHONPATH="$ROOT/code" \
    "$PY" -u code/token_sweep.py --model "$m" || echo "token sweep failed: $m"
done
