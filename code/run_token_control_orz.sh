#!/usr/bin/env bash
# Exploratory token/terminator control for ORZ-7B-LaSeR (code/token_control.py), MATH500 seed 0.
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
GPU_RUN="${GPU_RUN:-}"   # optional wrapper that takes a GPU for the command; empty = run directly
[ -s results/analysis/token_control_ORZ-7B-LaSeR.json ] && exit 0
$GPU_RUN env ${HF_HOME:+"HF_HOME=$HF_HOME"} HF_HUB_DISABLE_XET=1 PYTHONPATH="$ROOT/code" \
  "$PY" -u code/token_control.py --model ORZ-7B-LaSeR
