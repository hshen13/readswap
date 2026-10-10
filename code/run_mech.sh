#!/usr/bin/env bash
# Exploratory mechanism control (code/mech_control.py) on the seed-0 solutions of all five benchmarks, then the
# OctoThinker token sweep again with raw log-probability differences (median_abs_diff_lp); an older sweep file
# without them is kept as token_sweep_OctoThinker-3B-Short-LaSeR_v1.json.
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
for m in OctoThinker-3B-Short-LaSeR Qwen2.5-7B-LaSeR ORZ-7B-LaSeR; do
  [ -s "results/analysis/mech/${m}.jsonl" ] && continue
  $GPU_RUN env ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET=1 PYTHONPATH="$ROOT/code" \
    "$PY" -u code/mech_control.py --model "$m" || echo "mech control failed: $m"
done
if ! grep -q median_abs_diff_lp results/analysis/token_sweep_OctoThinker-3B-Short-LaSeR.json 2>/dev/null; then
  mv results/analysis/token_sweep_OctoThinker-3B-Short-LaSeR.json results/analysis/token_sweep_OctoThinker-3B-Short-LaSeR_v1.json
  $GPU_RUN env ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET=1 PYTHONPATH="$ROOT/code" \
    "$PY" -u code/token_sweep.py --model OctoThinker-3B-Short-LaSeR || echo "token sweep failed"
fi
echo "mech done $(date -u +%FT%TZ)"
