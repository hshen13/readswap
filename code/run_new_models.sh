#!/usr/bin/env bash
# Score-only training in Qwen3.5-4B and Gemma-4-12B (exploratory), the protocol of q3_pipeline.sh with seed 0:
# the base model writes eight solutions per problem once, a LoRA adapter learns only LaSeR's self-reward term on
# the training solutions, and the untrained and trained models read the same fixed test solutions.
# Test sets: the five benchmarks, AIME26 and HMMT26. Every step is skipped when its output exists.
# PYTHON_NEW runs the 2026 stack (requirements-2026.txt); PYTHON runs grading.
set -u
ulimit -c 0
ROOT="${READSWAP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
NPY="${PYTHON_NEW:-$PY}"
GPU_RUN="${GPU_RUN:-}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoids a FlashInfer kernel build with CUDA 12.4; sampling is unchanged

for model in qwen35_4b gemma4_12b; do
  Q="results/$model"
  export Q3_DIR="$ROOT/$Q"
  mkdir -p "$Q/data" "$Q/gen" "$Q/graded" "$Q/readout"
  [ -s "$Q/data/train.jsonl" ] || cp results/q3/data/train.jsonl "$Q/data/train.jsonl"
  if [ ! -s "$Q/gen/train.jsonl" ]; then
    $GPU_RUN "$NPY" -u "code/${model}_generate.py" --input "$Q/data/train.jsonl" --output "$Q/gen/train.jsonl" \
      --dataset deepmath || exit 1
  fi
  for ds in math500 amc23 aime24 aime25 OlympiadBench aime_2026 hmmt_feb_2026; do
    src="data/confirmatory/$ds/test.jsonl"
    [ -s "$src" ] || src="data/new_problems/$ds/test.jsonl"
    if [ ! -s "$Q/gen/$ds.jsonl" ]; then
      $GPU_RUN "$NPY" -u "code/${model}_generate.py" --input "$src" --output "$Q/gen/$ds.jsonl" --dataset "$ds" || exit 1
    fi
  done
  for f in train math500 amc23 aime24 aime25 OlympiadBench aime_2026 hmmt_feb_2026; do
    if [ ! -s "$Q/graded/$f.jsonl" ]; then
      "$PY" -u code/grade.py --input "$Q/gen/$f.jsonl" --output "$Q/graded/$f.jsonl" || exit 1
    fi
  done
  [ -s "$Q/model/train_meta.json" ] || $GPU_RUN "$NPY" -u "code/${model}_train.py" || exit 1
  [ -s "$Q/readout_meta.json" ] || $GPU_RUN "$NPY" -u "code/${model}_readout.py" || exit 1
  Q3_OUT="$ROOT/results/analysis/${model}_replication.json" "$PY" -u "code/${model}_analyze.py" || exit 1
done
echo "new models done"
