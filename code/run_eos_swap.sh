#!/usr/bin/env bash
# Exploratory exact-token EOS control (code/eos_swap_control.py), MATH500 seed 0, three checkpoints.
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
for m in OctoThinker-3B-Short-LaSeR Qwen2.5-7B-LaSeR ORZ-7B-LaSeR; do
  [ -s "results/analysis/eos_swap_${m}.json" ] && continue
  $GPU_RUN env ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET=1 PYTHONPATH="$ROOT/code" \
    "$PY" -u code/eos_swap_control.py --model "$m" || echo "eos swap failed: $m"
done
